# -*- coding: utf-8 -*-

# waifu2x.py by @marcan42 based on https://github.com/nagadomi/waifu2x
# MIT license, see https://github.com/nagadomi/waifu2x/blob/master/LICENSE

# 必要なモジュールのインポート
import json, sys, numpy as np # jsonサポート、システム、数値ユーティリティ(と思われる)
from scipy import misc, signal # scipyより、その他処理、信号処理
from PIL import Image # 画像ファイル取り扱いユーティリティ
# Model export script: https://mrcn.st/t/export_model.lua (needs a working waifu2x install)

#infile, outfile, 
modelpath = sys.argv[1:] # コマンドライン引数読み込み(入力ファイル、出力ファイル、モデルjsonファイルへのパス)

model = json.load(open(modelpath)) # jsonファイルのロード(jsonファイルに記述された構造をそのままメモリ上に展開していると思われる)
# モデルの1階層が持つ情報:
#   * nInputPlane : このモデルに入力するべき平面の数
#   * nOutputPlane : このモデルから出力されるべき平面の数
#   * weight : 入力平面に対して畳み込み演算を行う重み行列の集合
#     - nInputPlane個の重み行列をパックしたものがnOutputPlane個入っている。行列の要素は全てfloat値。
#     - 例えば、初回は1個の重み行列をパックしたものが32個、2階層目は32個の重み行列をパックしたものが32個(つまり行列の数は1024個)入っている。
#     - 1つの重み行列は全て3x3行列となっている。従って、1回畳み込みを行う度に平面の大きさは上下左右1要素ずつ小さくなる。
#     - 計算時のアンカー(畳み込みの中心とする行列の位置)はおそらく(2,2)であると思われる(そうでないと、出力平面が変な方向に移動してしまうため)。
#   * bias : 出力平面一つに対して(全要素に加算することにより)掛けるバイアス値
#     - float値がnOutputPlaneの数だけ用意されている。
#   * kW,kH : 未使用値(重み行列1つの大きさであると思われる。配布されているモデルの場合は全て3x3である)

im = Image.open(infile).convert("YCbCr") # 入力ファイルの読み込み -> YCbCr色空間への変換
im = misc.fromimage(im.resize((2*im.size[0], 2*im.size[1]), resample=Image.NEAREST)).astype("float32")
# 入力画像を2倍の幅・高さにNearestNeighbor法でリサイズした後、それをscipyで取り扱い可能な行列表現にし、
# その要素の型を32bit-floatにする

planes = [np.pad(im[:,:,0], len(model), "edge") / 255.0]
# 画像データの周りに、画像の端をコピーする形で、モデルの大きさ(核の行列の大きさ)分だけパッドを入れ、0~1の間でクリップする
# このplanesは輝度情報のみを取り出している(!)
# オリジナルのwaifu2xも、reconstruct時に入力をYUV色空間に変換した後、Yのみを取り出して処理している

count = sum(step["nInputPlane"] * step["nOutputPlane"] for step in model)# 畳み込み演算の必要回数を計算
# つまり、countの数だけ入力平面に対する重み行列の畳み込みが行われる。
progress = 0

for step in model: # ループ:ステップ(1つのモデル階層) 始め
    assert step["nInputPlane"] == len(planes) 
    # このステップのモデルに定義された入力平面の数と実際の入力平面の数は一致していなければならない
    assert step["nOutputPlane"] == len(step["weight"]) == len(step["bias"])
    # モデルの出力平面はモデルの重み行列集合の数とそのバイアスの数と一致していなければならない
    # つまり、各ステップの重み行列集合の数とそのバイアスの数だけ、そのステップによって平面が出力される
    
    o_planes = [] # 出力平面の格納場所を初期化
    
    for bias, weights in zip(step["bias"], step["weight"]): # ループ:バイアス&重み行列集合 始め
        partial = None # partialをNone(null値)に初期化
        for ip, kernel in zip(planes, weights): # ループ:入力平面&核(重み行列) 始め
            p = signal.fftconvolve(ip, np.float32(kernel), "valid") # 入力平面に対して核を畳み込み演算
            if partial is None:
                partial = p # 最初の畳み込み演算の結果を代入
            else:
                partial += p # 畳み込み演算の結果を加算したものを代入
                # したがって、partialにはこのステップにおける全ての入力平面に対する重み行列の畳み込み演算の結果の総和が入る
            progress += 1
            sys.stderr.write("\r%.1f%%..." % (100 * progress / float(count)))
        # ループ:入力平面&核 終わり
        partial += np.float32(bias) # 計算したpartialにバイアスを加算(バイアスは1つの数値なので、partial全体にバイアスがかかる)
        # ここまでが1つの出力平面の生成処理
        o_planes.append(partial) # 出力平面の集合にpartialを加える
    # ループ:バイアス&重み 終わり
    planes = [np.maximum(p, 0) + 0.1 * np.minimum(p, 0) for p in o_planes]
    # 出力平面(を表す行列)に存在する負の値を0.1倍する
    
    # この時点で、出力平面このステップにおけるnOutputPlaneの数だけ生成される

# ループ:ステップ 終わり

assert len(planes) == 1 # 最後のステップにおける出力平面は1つでなければならない
im[:,:,0] = np.clip(planes[0], 0, 1) * 255
# 得られた出力平面の全要素を0~1にクリップした後、
misc.toimage(im, mode="YCbCr").convert("RGB").save(outfile)
sys.stderr.write("Done\n")
