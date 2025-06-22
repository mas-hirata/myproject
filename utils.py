# 複数画像の高さを最小値に揃えてリサイズ
# リサイズ後の画像を横に並べた大きなキャンバスを生成
# 左から順番に貼り付け、１枚の長い画像に結合

from PIL import Image
im1 = Image.open('/kaggle/input/byu-d-301/F1_curve.png')
im2 = Image.open('/kaggle/input/byu-d-301/PR_curve.png')
im3 = Image.open('/kaggle/input/byu-d-301/dfl_loss_curve.png')
def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
    min_height = min(im.height for im in im_list)
    im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height),resample=resample)
                      for im in im_list]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst
get_concat_h_multi_resize([im3, im2, im1])