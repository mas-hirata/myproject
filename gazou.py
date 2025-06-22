# 目的：画像の「暗すぎ／明るすぎ」を抑えて、見やすいコントラストに調整する
# 手順：
#  画素値の下から2%／上から2%に当たる値 p2／p98 を取得（外れ値を除く）
#  np.clip で画素値を [p2, p98] に制限
#  0–255 の範囲にリスケールして uint8 型に変換


def normalize_slice(slice_data):
    """
    Normalize slice data using 2nd and 98th percentiles for better contrast
    """
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    clipped_data = np.clip(slice_data, p2, p98)
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    return np.uint8(normalized)


# 目的：次の推論バッチに使う画像を、前もって（並行して）読み込んでおく
# 流れ：
#  OpenCV (cv2.imread) で読めなければ PIL (Image.open) にフォールバック
#  リストに追加してまとめて返す


def preload_image_batch(file_paths):
    """Preload a batch of images to CPU memory"""
    images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is None:
            # OpenCVで読めない場合はPILで再トライ
            img = np.array(Image.open(path))
        images.append(img)
    return images

# 1つのトモグラム（断層像セット）に対して「バッチ処理＋並列化＋GPUストリーム」で 
# YOLO 推論を行い、「最も確信度の高いモーター位置」を返す関数
def process_tomogram(tomo_id, model, index=0, total=1):
    """
    Process a single tomogram and return the most confident motor detection
    """
    # print(f"Processing tomogram {tomo_id} ({index}/{total})")
    
    # test_dir/tomo_id フォルダ内の .jpg ファイル名をすべて取得し、ソートしてリスト化
    tomo_dir = os.path.join(test_dir, tomo_id)
    slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])

    # CONCENTRATION：全スライスのうち「何割を使うか」を示す（例 1.0＝100%, 0.5＝半分）
    # np.linspace で等間隔にインデックスを取り、round して整数化→間引きしたスライスリストに更新
    # これにより「処理量を減らしたい」「ざっと全体をざっくり見る」ような設定が可能

    selected_indices = np.linspace(0, len(slice_files)-1, int(len(slice_files) * CONCENTRATION))
    selected_indices = np.round(selected_indices).astype(int)
    slice_files = [slice_files[i] for i in selected_indices]
    
    # ストリーム（torch.cuda.Stream）：CUDA の非同期実行コンテキスト
    # 複数ストリームを用意してバッチ内をさらに並列化し、GPU の空きリソースを効率利用
    # ストリーム数は BATCH_SIZE と最大４の小さい方

    if device.startswith('cuda'):
        全サブバッチ終了後の GPU 同期 = [torch.cuda.Stream() for _ in range(min(4, BATCH_SIZE))]
    else:
        全サブバッチ終了後の GPU 同期 = [None]
    
    # all_detections：このトモグラム内の全スライスから得られた検出結果を累積
    # next_batch_thread：次のバッチ画像を先読みするスレッドを保持
    all_detections = []
    next_batch_thread = None
    next_batch_images = None
    
    # バッチ単位でループ
    # 　batch_start を 0 ～ スライス数 まで BATCH_SIZE 間隔で刻む。
    # 　前回立ち上げたプリロードスレッド (next_batch_thread) があれば .join() して完了を待機。
    for batch_start in range(0, len(slice_files), BATCH_SIZE):
        # Wait for previous preload thread if it exists
        if next_batch_thread is not None:
            next_batch_thread.join()
            next_batch_images = None
        # batch_files：今すぐ推論するファイルリスト
        batch_end = min(batch_start + BATCH_SIZE, len(slice_files))
        batch_files = slice_files[batch_start:batch_end]
        
        # next_batch_files：次ループで使うファイルリスト
        next_batch_start = batch_end
        next_batch_end = min(next_batch_start + BATCH_SIZE, len(slice_files))
        next_batch_files = slice_files[next_batch_start:next_batch_end] if next_batch_start < len(slice_files) else []
        

        # threading.Thread を使い、別スレッドで preload_image_batch（OpenCV/PIL での画像読み込み）を実行
        # メインスレッドはその間に現在バッチを処理
        if next_batch_files:
            next_batch_paths = [os.path.join(tomo_dir, f) for f in next_batch_files]
            next_batch_thread = threading.Thread(target=preload_image_batch, args=(next_batch_paths,))
            next_batch_thread.start()
        else:
            next_batch_thread = None
        
        # サブバッチに分割してストリームごとに推論
        # batch_files: 今回まとめて処理するスライス画像ファイルのリスト
        # 全サブバッチ終了後の GPU 同期: 用意した CUDA ストリームのリスト（GPU を使わない場合は [None]）
        # Python のリスト batch_files を、streams の数だけ均等に分割。
        # たとえばバッチに 8 枚の画像があってストリーム数が 2 なら、4 枚ずつに分かれます。
        # これを sub_batches と呼びます。

        # i % len(全サブバッチ終了後の GPU 同期) で、サブバッチ i に割り当てるストリームを循環的に選びます。
        # こうすると「ストリーム１にはサブバッチ１と３と５…」「ストリーム２にはサブバッチ２と４…」のように分散されます。
        sub_batches = np.array_split(batch_files, len(全サブバッチ終了後の GPU 同期))
        sub_batch_results = []
        
        for i, sub_batch in enumerate(sub_batches):
            if len(sub_batch) == 0:
                continue
                
            stream = 全サブバッチ終了後の GPU 同期[i % len(全サブバッチ終了後の GPU 同期)]
            with torch.cuda.stream(stream) if stream and device.startswith('cuda') else nullcontext():
                # Process sub-batch
                sub_batch_paths = [os.path.join(tomo_dir, slice_file) for slice_file in sub_batch]
                sub_batch_slice_nums = [int(slice_file.split('_')[1].split('.')[0]) for slice_file in sub_batch]
                
                # GPUProfiler は「ここからここまで GPU の処理時間を測る」ための仕掛け。
                # model(sub_batch_paths) で YOLO モデルに一気に複数画像を渡し、物体検出を行います。
                # 戻り値 sub_results は、各画像につき 1 件の「結果オブジェクト」が入ったリストです。
                with GPUProfiler(f"Inference batch {i+1}/{len(sub_batches)}"):
                    sub_results = model(sub_batch_paths, verbose=False)
                
                # 推論結果の後処理
                    # result.boxes：検出された矩形（バウンディングボックス）情報のまとまり
                    # boxes.conf：各ボックスごとの「この予測が正解の確信度」
                    # boxes.xyxy：左上 (x1,y1)／右下 (x2,y2) の座標
                    # 確信度が CONFIDENCE_THRESHOLD 以上なら採用
                    # ボックスの中心を (x1+x2)/2, (y1+y2)/2 で計算
                    # スライス番号 z とあわせて all_detections に格納
                    
                for j, result in enumerate(sub_results):
                    if len(result.boxes) > 0:
                        boxes = result.boxes
                        for box_idx, confidence in enumerate(boxes.conf):
                            if confidence >= CONFIDENCE_THRESHOLD:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = boxes.xyxy[box_idx].cpu().numpy()
                                
                                # Calculate center coordinates
                                x_center = (x1 + x2) / 2
                                y_center = (y1 + y2) / 2
                                
                                # Store detection with 3D coordinates
                                all_detections.append({
                                    'z': round(sub_batch_slice_nums[j]),
                                    'y': round(y_center),
                                    'x': round(x_center),
                                    'confidence': float(confidence)
                                })
        
        # 全サブバッチ終了後の GPU 同期
        if device.startswith('cuda'):
            torch.cuda()
    

    # all_detections：このトモグラム全体から集めた「3D座標＋信頼度」のリスト
    # 3D NMS で近接検出をマージ
    # 信頼度順にソート
    # 一番信頼度の高いものだけを最終結果として返却
    # もし検出が一つもなければ -1, -1, -1 を返す
    if next_batch_thread is not None:
        next_batch_thread.join()
    
    final_detections = perform_3d_nms(all_detections, NMS_IOU_THRESHOLD)
    
    final_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    if not final_detections:
        return {
            'tomo_id': tomo_id,
            'Motor axis 0': -1,
            'Motor axis 1': -1,
            'Motor axis 2': -1
        }
    
    best_detection = final_detections[0]
    
    return {
        'tomo_id': tomo_id,
        'Motor axis 0': round(best_detection['z']),
        'Motor axis 1': round(best_detection['y']),
        'Motor axis 2': round(best_detection['x'])
    }

# 📋 目的
    # 同じモーターを複数スライスで検出してしまったときに、「近接しているもの」をまとめて一つに絞り込む。

# 🔧 入力
    # detections：
    # 各要素が { 'z':…, 'y':…, 'x':…, 'confidence':… } の辞書のリスト
    # z, y, x はそれぞれスライス番号・行・列の位置、confidence は検出確信度
    # iou_threshold：
    # 「どれだけ近ければ同一とみなすか」を決める距離のしきい値係数

def perform_3d_nms(detections, iou_threshold):
    """
    Perform 3D Non-Maximum Suppression on detections to merge nearby motors
    """
    # 検出が一つもなければ、そのまま空リストを返します。
    if not detections:
        return []
    
    # 一番「自信のある」検出から順に処理を始めるために、並び替え。
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # Define 3D distance function
    def distance_3d(d1, d2):
        return np.sqrt((d1['z'] - d2['z'])**2 + 
                       (d1['y'] - d2['y'])**2 + 
                       (d1['x'] - d2['x'])**2)

    # 「ボックスサイズ × 差し引き閾値係数」で、”近い” とみなす最大距離を決定。
    box_size = 24  # Same as annotation box size
    distance_threshold = box_size * iou_threshold
    

    # 最高確信度の検出をひとつ選ぶ
    # それを「確定」リストに入れる
    # 「確定」した位置から distance_threshold 以下の距離にある候補をすべて除く
    # 残りで同じ処理を繰り返す
    final_detections = []
    while detections:
        best_detection = detections.pop(0)
        final_detections.append(best_detection)
        
        detections = [d for d in detections if distance_3d(d, best_detection) > distance_threshold]
    
    return final_detections


# 📋 目的
#   トモグラムの画像ファイルが正しく読めるか、いくつかの方法で試し読みして問題がないかを確認する

# 🔧 入力
#   tomo_id：
#   調べたいトモグラムフォルダ名
def debug_image_loading(tomo_id):
    """
    Debug function to check image loading
    """
    tomo_dir = os.path.join(test_dir, tomo_id)
    slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
    
    if not slice_files:
        print(f"No image files found in {tomo_dir}")
        return

    # リストの真ん中にある画像を代表サンプルとして選択。    
    sample_file = slice_files[len(slice_files)//2]  # Middle slice
    img_path = os.path.join(tomo_dir, sample_file)
    

    # PIL → Image.open → 配列化
    # OpenCV → cv2.imread
    # OpenCV＋色変換 → RGBに直す
    # いずれかで例外が出たら、どの方法で失敗したかを出力。
    try:
        # Method 1: PIL
        img_pil = Image.open(img_path)
        img_array_pil = np.array(img_pil)
        
        # Method 2: OpenCV
        img_cv2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # print(f"OpenCV Image shape: {img_cv2.shape}, dtype: {img_cv2.dtype}")
        
        # Method 3: Convert to RGB
        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        
    # Also test with YOLO's built-in loader
    try:
        test_model = YOLO(model_path)
        test_results = test_model([img_path], verbose=False)
        # print("YOLO model successfully processed the test image")
    except Exception as e:
        print(f"Error with YOLO processing: {e}")

# テスト用トモグラム全体を一気に処理し、「提出用 CSV」を作成するメインルーチン
def generate_submission():
    """
    Main function to generate the submission file
    """
    test_tomos = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    total_tomos = len(test_tomos)
    
    if test_tomos:
        debug_image_loading(test_tomos[0])
    
    # GPU を使う場合、PyTorch が保持している不要な GPU メモリをクリア
    # これで以降の処理に最大限メモリを割り当てられるように準備
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = YOLO(model_path)
    model.to(device)
    
    GPU 最適化

    # model.fuse()
    #   畳み込み層（Conv）＋バッチ正規化層（BatchNorm）を結合し、推論速度を向上
    # model.model.half()
    #   GPU が Ampere 以降（Volta 相当）であれば、**半精度浮動小数点（FP16）**にモデルを変換しメモリ使用量／高速化を図る
    if device.startswith('cuda'):
        model.fuse()
        
        if torch.cuda.get_device_capability(0)[0] >= 7:  # Volta or newer
            model.model.half()
    
    # Process tomograms with parallelization
    results = []
    motors_found = 0


    # ThreadPoolExecutor を使って 並列処理プール を開く
    # max_workers=1 なので実際は一度に１タスクずつですが、他の設定なら複数同時実行も可能
    # テスト用トモグラムすべてについて、
    # process_tomogram 関数を「非同期タスク」としてキューに投入（submit）
    # 戻ってくる future（将来の結果オブジェクト）と tomo_id を対応づけて保存
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_tomo = {}
        
        for i, tomo_id in enumerate(test_tomos, 1):
            future = executor.submit(process_tomogram, tomo_id, model, i, total_tomos)
            future_to_tomo[future] = tomo_id
        
        # Process completed futures as they complete
        for future in future_to_tomo:
            tomo_id = future_to_tomo[future]
            try:
                # Clear CUDA cache between tomograms
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                result = future.result()
                results.append(result)
                
                # future_to_tomo に登録したすべての future についてループ
                # future.result() で process_tomogram の返り値を取得（完了を待つ）
                # 結果を results リストに追加
                # Motor axis 0 が -1 でなければ「検出あり」と判断しカウンタを増加＆ログ出力
                # 処理率（何件中何件検出できたか）を随時表示
                # 例外が出た場合もサイレントに失敗せず、座標を -1 で埋めておく
                has_motor = not pd.isna(result['Motor axis 0'])
                if has_motor:
                    motors_found += 1
                    print(f"Motor found in {tomo_id} at position: "
                          f"z={result['Motor axis 0']}, y={result['Motor axis 1']}, x={result['Motor axis 2']}")
                else:
                    print(f"No motor detected in {tomo_id}")
                    
                print(f"Current detection rate: {motors_found}/{len(results)} ({motors_found/len(results)*100:.1f}%)")
            
            except Exception as e:
                print(f"Error processing {tomo_id}: {e}")
                # Create a default entry for failed tomograms
                results.append({
                    'tomo_id': tomo_id,
                    'Motor axis 0': -1,
                    'Motor axis 1': -1,
                    'Motor axis 2': -1
                })
    
    # Create submission dataframe
    submission_df = pd.DataFrame(results)
    
    # Ensure proper column order
    submission_df = submission_df[['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]
    
    # Save the submission file
    submission_df.to_csv(submission_path, index=False)
    print("="*50)
    print("= Submission preview:")
    print("="*50)
    print(submission_df.head())
    return submission_df
    