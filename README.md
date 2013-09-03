SdAE
====

stacked denoising autoencoder.

とりあえずubuntuでは動く.

Eigen http://eigen.tuxfamily.org/index.php?title=Main_Page を使用.


モデルは

pretrainingにdenoising autoencoder

http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf

出力レイヤには二乗正則化のロジスティクス回帰

を使用


データの読み込み等は適時改変して下さい.
