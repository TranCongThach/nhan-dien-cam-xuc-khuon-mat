import re
import matplotlib.pyplot as plt

# =====================================================
# 1) COPY NGUYÊN LOG BẠN GỬI VÀ DÁN VÀO BIẾN log_text
# =====================================================

log_text = r"""
Epoch 1/60
1345/1345 [==============================] - ETA: 0s - loss: 2.4045 - accuracy: 0.2050   
Epoch 1: val_loss improved from inf to 1.88077, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 452s 335ms/step - loss: 2.4045 - accuracy: 0.2050 - val_loss: 1.8808 - val_accuracy: 0.3150 - lr: 3.0000e-04
Epoch 2/60
1345/1345 [==============================] - ETA: 0s - loss: 2.0185 - accuracy: 0.2736  
Epoch 2: val_loss improved from 1.88077 to 1.73902, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 425s 316ms/step - loss: 2.0185 - accuracy: 0.2736 - val_loss: 1.7390 - val_accuracy: 0.3969 - lr: 3.0000e-04
Epoch 3/60
1345/1345 [==============================] - ETA: 0s - loss: 1.8395 - accuracy: 0.3461   
Epoch 3: val_loss improved from 1.73902 to 1.73120, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 448s 333ms/step - loss: 1.8395 - accuracy: 0.3461 - val_loss: 1.7312 - val_accuracy: 0.4152 - lr: 3.0000e-04
Epoch 4/60
1345/1345 [==============================] - ETA: 0s - loss: 1.7236 - accuracy: 0.3979  
Epoch 4: val_loss improved from 1.73120 to 1.62510, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 416s 309ms/step - loss: 1.7236 - accuracy: 0.3979 - val_loss: 1.6251 - val_accuracy: 0.4653 - lr: 3.0000e-04
Epoch 5/60
1345/1345 [==============================] - ETA: 0s - loss: 1.6363 - accuracy: 0.4439  
Epoch 5: val_loss improved from 1.62510 to 1.53477, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 407s 303ms/step - loss: 1.6363 - accuracy: 0.4439 - val_loss: 1.5348 - val_accuracy: 0.4986 - lr: 3.0000e-04
Epoch 6/60
1345/1345 [==============================] - ETA: 0s - loss: 1.5734 - accuracy: 0.4782  
Epoch 6: val_loss improved from 1.53477 to 1.49542, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 396s 294ms/step - loss: 1.5734 - accuracy: 0.4782 - val_loss: 1.4954 - val_accuracy: 0.5110 - lr: 3.0000e-04
Epoch 7/60
1345/1345 [==============================] - ETA: 0s - loss: 1.5298 - accuracy: 0.4977  
Epoch 7: val_loss improved from 1.49542 to 1.42473, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 409s 304ms/step - loss: 1.5298 - accuracy: 0.4977 - val_loss: 1.4247 - val_accuracy: 0.5477 - lr: 3.0000e-04
Epoch 8/60
1345/1345 [==============================] - ETA: 0s - loss: 1.4945 - accuracy: 0.5109  
Epoch 8: val_loss did not improve from 1.42473
1345/1345 [==============================] - 425s 316ms/step - loss: 1.4945 - accuracy: 0.5109 - val_loss: 1.4604 - val_accuracy: 0.5290 - lr: 3.0000e-04
Epoch 9/60
1345/1345 [==============================] - ETA: 0s - loss: 1.4652 - accuracy: 0.5265  
Epoch 9: val_loss improved from 1.42473 to 1.36780, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 452s 336ms/step - loss: 1.4652 - accuracy: 0.5265 - val_loss: 1.3678 - val_accuracy: 0.5707 - lr: 3.0000e-04
Epoch 10/60
1345/1345 [==============================] - ETA: 0s - loss: 1.4385 - accuracy: 0.5458  
Epoch 10: val_loss improved from 1.36780 to 1.36318, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 436s 324ms/step - loss: 1.4385 - accuracy: 0.5458 - val_loss: 1.3632 - val_accuracy: 0.5739 - lr: 3.0000e-04
Epoch 11/60
1345/1345 [==============================] - ETA: 0s - loss: 1.4282 - accuracy: 0.5451  
Epoch 11: val_loss improved from 1.36318 to 1.34891, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 450s 334ms/step - loss: 1.4282 - accuracy: 0.5451 - val_loss: 1.3489 - val_accuracy: 0.5755 - lr: 3.0000e-04
Epoch 12/60
1345/1345 [==============================] - ETA: 0s - loss: 1.4032 - accuracy: 0.5579  
Epoch 12: val_loss improved from 1.34891 to 1.31526, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 431s 320ms/step - loss: 1.4032 - accuracy: 0.5579 - val_loss: 1.3153 - val_accuracy: 0.5903 - lr: 3.0000e-04
Epoch 13/60
1345/1345 [==============================] - ETA: 0s - loss: 1.3912 - accuracy: 0.5636  
Epoch 13: val_loss improved from 1.31526 to 1.31087, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 397s 295ms/step - loss: 1.3912 - accuracy: 0.5636 - val_loss: 1.3109 - val_accuracy: 0.6011 - lr: 3.0000e-04
Epoch 14/60
1345/1345 [==============================] - ETA: 0s - loss: 1.3755 - accuracy: 0.5739  
Epoch 14: val_loss improved from 1.31087 to 1.28679, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 314s 234ms/step - loss: 1.3755 - accuracy: 0.5739 - val_loss: 1.2868 - val_accuracy: 0.6119 - lr: 3.0000e-04
Epoch 15/60
1345/1345 [==============================] - ETA: 0s - loss: 1.3656 - accuracy: 0.5796  
Epoch 15: val_loss did not improve from 1.28679
1345/1345 [==============================] - 315s 234ms/step - loss: 1.3656 - accuracy: 0.5796 - val_loss: 1.3145 - val_accuracy: 0.5953 - lr: 3.0000e-04
Epoch 16/60
1345/1345 [==============================] - ETA: 0s - loss: 1.3520 - accuracy: 0.5863  
Epoch 16: val_loss improved from 1.28679 to 1.28073, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 310s 230ms/step - loss: 1.3520 - accuracy: 0.5863 - val_loss: 1.2807 - val_accuracy: 0.6102 - lr: 3.0000e-04
Epoch 17/60
1345/1345 [==============================] - ETA: 0s - loss: 1.3430 - accuracy: 0.5913  
Epoch 17: val_loss did not improve from 1.28073
1345/1345 [==============================] - 315s 234ms/step - loss: 1.3430 - accuracy: 0.5913 - val_loss: 1.3083 - val_accuracy: 0.6048 - lr: 3.0000e-04
Epoch 18/60
1345/1345 [==============================] - ETA: 0s - loss: 1.3321 - accuracy: 0.5977  
Epoch 18: val_loss did not improve from 1.28073
1345/1345 [==============================] - 309s 230ms/step - loss: 1.3321 - accuracy: 0.5977 - val_loss: 1.3051 - val_accuracy: 0.6000 - lr: 3.0000e-04
Epoch 19/60
1345/1345 [==============================] - ETA: 0s - loss: 1.3258 - accuracy: 0.6024  
Epoch 19: val_loss improved from 1.28073 to 1.26545, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 328s 244ms/step - loss: 1.3258 - accuracy: 0.6024 - val_loss: 1.2654 - val_accuracy: 0.6193 - lr: 3.0000e-04
Epoch 20/60
1345/1345 [==============================] - ETA: 0s - loss: 1.3141 - accuracy: 0.6060  
Epoch 20: val_loss improved from 1.26545 to 1.25566, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 352s 262ms/step - loss: 1.3141 - accuracy: 0.6060 - val_loss: 1.2557 - val_accuracy: 0.6296 - lr: 3.0000e-04
Epoch 21/60
1345/1345 [==============================] - ETA: 0s - loss: 1.3058 - accuracy: 0.6095  
Epoch 21: val_loss improved from 1.25566 to 1.25052, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 371s 276ms/step - loss: 1.3058 - accuracy: 0.6095 - val_loss: 1.2505 - val_accuracy: 0.6256 - lr: 3.0000e-04
Epoch 22/60
1345/1345 [==============================] - ETA: 0s - loss: 1.3036 - accuracy: 0.6144  
Epoch 22: val_loss did not improve from 1.25052
1345/1345 [==============================] - 354s 263ms/step - loss: 1.3036 - accuracy: 0.6144 - val_loss: 1.2786 - val_accuracy: 0.6112 - lr: 3.0000e-04
Epoch 23/60
1345/1345 [==============================] - ETA: 0s - loss: 1.2988 - accuracy: 0.6133  
Epoch 23: val_loss did not improve from 1.25052
1345/1345 [==============================] - 357s 265ms/step - loss: 1.2988 - accuracy: 0.6133 - val_loss: 1.2679 - val_accuracy: 0.6247 - lr: 3.0000e-04
Epoch 24/60
1345/1345 [==============================] - ETA: 0s - loss: 1.2907 - accuracy: 0.6184  
Epoch 24: val_loss did not improve from 1.25052

Epoch 24: ReduceLROnPlateau reducing learning rate to 0.0001500000071246177.
1345/1345 [==============================] - 357s 266ms/step - loss: 1.2907 - accuracy: 0.6184 - val_loss: 1.2723 - val_accuracy: 0.6225 - lr: 3.0000e-04       
Epoch 25/60
1345/1345 [==============================] - ETA: 0s - loss: 1.2552 - accuracy: 0.6374  
Epoch 25: val_loss improved from 1.25052 to 1.23405, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 355s 264ms/step - loss: 1.2552 - accuracy: 0.6374 - val_loss: 1.2341 - val_accuracy: 0.6359 - lr: 1.5000e-04
Epoch 26/60
1345/1345 [==============================] - ETA: 0s - loss: 1.2507 - accuracy: 0.6392  
Epoch 26: val_loss did not improve from 1.23405
1345/1345 [==============================] - 356s 265ms/step - loss: 1.2507 - accuracy: 0.6392 - val_loss: 1.2447 - val_accuracy: 0.6304 - lr: 1.5000e-04
Epoch 27/60
1345/1345 [==============================] - ETA: 0s - loss: 1.2390 - accuracy: 0.6440  
Epoch 27: val_loss improved from 1.23405 to 1.23042, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 362s 269ms/step - loss: 1.2390 - accuracy: 0.6440 - val_loss: 1.2304 - val_accuracy: 0.6371 - lr: 1.5000e-04
Epoch 28/60
1345/1345 [==============================] - ETA: 0s - loss: 1.2361 - accuracy: 0.6448  
Epoch 28: val_loss improved from 1.23042 to 1.22762, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 355s 264ms/step - loss: 1.2361 - accuracy: 0.6448 - val_loss: 1.2276 - val_accuracy: 0.6394 - lr: 1.5000e-04
Epoch 29/60
1345/1345 [==============================] - ETA: 0s - loss: 1.2249 - accuracy: 0.6547  
Epoch 29: val_loss improved from 1.22762 to 1.22058, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 353s 263ms/step - loss: 1.2249 - accuracy: 0.6547 - val_loss: 1.2206 - val_accuracy: 0.6479 - lr: 1.5000e-04
Epoch 30/60
1345/1345 [==============================] - ETA: 0s - loss: 1.2185 - accuracy: 0.6548  
Epoch 30: val_loss did not improve from 1.22058
1345/1345 [==============================] - 354s 263ms/step - loss: 1.2185 - accuracy: 0.6548 - val_loss: 1.2427 - val_accuracy: 0.6387 - lr: 1.5000e-04
Epoch 31/60
1345/1345 [==============================] - ETA: 0s - loss: 1.2164 - accuracy: 0.6516  
Epoch 31: val_loss improved from 1.22058 to 1.20824, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 371s 275ms/step - loss: 1.2164 - accuracy: 0.6516 - val_loss: 1.2082 - val_accuracy: 0.6503 - lr: 1.5000e-04
Epoch 32/60
1345/1345 [==============================] - ETA: 0s - loss: 1.2076 - accuracy: 0.6598  
Epoch 32: val_loss improved from 1.20824 to 1.20087, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 363s 270ms/step - loss: 1.2076 - accuracy: 0.6598 - val_loss: 1.2009 - val_accuracy: 0.6530 - lr: 1.5000e-04
Epoch 33/60
1345/1345 [==============================] - ETA: 0s - loss: 1.2009 - accuracy: 0.6588  
Epoch 33: val_loss did not improve from 1.20087
1345/1345 [==============================] - 363s 270ms/step - loss: 1.2009 - accuracy: 0.6588 - val_loss: 1.2393 - val_accuracy: 0.6426 - lr: 1.5000e-04
Epoch 34/60
1345/1345 [==============================] - ETA: 0s - loss: 1.2042 - accuracy: 0.6628  
Epoch 34: val_loss did not improve from 1.20087
1345/1345 [==============================] - 357s 266ms/step - loss: 1.2042 - accuracy: 0.6628 - val_loss: 1.2105 - val_accuracy: 0.6470 - lr: 1.5000e-04
Epoch 35/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1919 - accuracy: 0.6686  
Epoch 35: val_loss did not improve from 1.20087

Epoch 35: ReduceLROnPlateau reducing learning rate to 7.500000356230885e-05.
1345/1345 [==============================] - 356s 265ms/step - loss: 1.1919 - accuracy: 0.6686 - val_loss: 1.2131 - val_accuracy: 0.6496 - lr: 1.5000e-04       
Epoch 36/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1729 - accuracy: 0.6732  
Epoch 36: val_loss did not improve from 1.20087
1345/1345 [==============================] - 356s 264ms/step - loss: 1.1729 - accuracy: 0.6732 - val_loss: 1.2137 - val_accuracy: 0.6491 - lr: 7.5000e-05
Epoch 37/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1683 - accuracy: 0.6787  
Epoch 37: val_loss improved from 1.20087 to 1.19928, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 376s 280ms/step - loss: 1.1683 - accuracy: 0.6787 - val_loss: 1.1993 - val_accuracy: 0.6549 - lr: 7.5000e-05
Epoch 38/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1630 - accuracy: 0.6801  
Epoch 38: val_loss improved from 1.19928 to 1.19641, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 361s 269ms/step - loss: 1.1630 - accuracy: 0.6801 - val_loss: 1.1964 - val_accuracy: 0.6607 - lr: 7.5000e-05
Epoch 39/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1583 - accuracy: 0.6788  
Epoch 39: val_loss improved from 1.19641 to 1.18889, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 354s 263ms/step - loss: 1.1583 - accuracy: 0.6788 - val_loss: 1.1889 - val_accuracy: 0.6582 - lr: 7.5000e-05
Epoch 40/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1569 - accuracy: 0.6821  
Epoch 40: val_loss did not improve from 1.18889
1345/1345 [==============================] - 374s 278ms/step - loss: 1.1569 - accuracy: 0.6821 - val_loss: 1.1965 - val_accuracy: 0.6543 - lr: 7.5000e-05
Epoch 41/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1480 - accuracy: 0.6874  
Epoch 41: val_loss did not improve from 1.18889
1345/1345 [==============================] - 390s 290ms/step - loss: 1.1480 - accuracy: 0.6874 - val_loss: 1.1982 - val_accuracy: 0.6561 - lr: 7.5000e-05
Epoch 42/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1441 - accuracy: 0.6879  
Epoch 42: val_loss did not improve from 1.18889

Epoch 42: ReduceLROnPlateau reducing learning rate to 3.7500001781154424e-05.
1345/1345 [==============================] - 365s 271ms/step - loss: 1.1441 - accuracy: 0.6879 - val_loss: 1.1921 - val_accuracy: 0.6543 - lr: 7.5000e-05       
Epoch 43/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1320 - accuracy: 0.6957  
Epoch 43: val_loss did not improve from 1.18889
1345/1345 [==============================] - 365s 271ms/step - loss: 1.1320 - accuracy: 0.6957 - val_loss: 1.1906 - val_accuracy: 0.6599 - lr: 3.7500e-05
Epoch 44/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1270 - accuracy: 0.6933  
Epoch 44: val_loss improved from 1.18889 to 1.18119, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 361s 269ms/step - loss: 1.1270 - accuracy: 0.6933 - val_loss: 1.1812 - val_accuracy: 0.6607 - lr: 3.7500e-05
Epoch 45/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1260 - accuracy: 0.6993  
Epoch 45: val_loss did not improve from 1.18119
1345/1345 [==============================] - 361s 269ms/step - loss: 1.1260 - accuracy: 0.6993 - val_loss: 1.1892 - val_accuracy: 0.6599 - lr: 3.7500e-05
Epoch 46/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1232 - accuracy: 0.7006  
Epoch 46: val_loss improved from 1.18119 to 1.17713, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 357s 265ms/step - loss: 1.1232 - accuracy: 0.7006 - val_loss: 1.1771 - val_accuracy: 0.6659 - lr: 3.7500e-05
Epoch 47/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1230 - accuracy: 0.6989  
Epoch 47: val_loss did not improve from 1.17713
1345/1345 [==============================] - 367s 273ms/step - loss: 1.1230 - accuracy: 0.6989 - val_loss: 1.1777 - val_accuracy: 0.6610 - lr: 3.7500e-05
Epoch 48/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1203 - accuracy: 0.6965  
Epoch 48: val_loss did not improve from 1.17713
1345/1345 [==============================] - 361s 268ms/step - loss: 1.1203 - accuracy: 0.6965 - val_loss: 1.1801 - val_accuracy: 0.6618 - lr: 3.7500e-05
Epoch 49/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1143 - accuracy: 0.7034  
Epoch 49: val_loss improved from 1.17713 to 1.17414, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 361s 268ms/step - loss: 1.1143 - accuracy: 0.7034 - val_loss: 1.1741 - val_accuracy: 0.6622 - lr: 3.7500e-05
Epoch 50/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1193 - accuracy: 0.6987  
Epoch 50: val_loss improved from 1.17414 to 1.16846, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 363s 270ms/step - loss: 1.1193 - accuracy: 0.6987 - val_loss: 1.1685 - val_accuracy: 0.6695 - lr: 3.7500e-05
Epoch 51/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1123 - accuracy: 0.7044  
Epoch 51: val_loss did not improve from 1.16846
1345/1345 [==============================] - 380s 283ms/step - loss: 1.1123 - accuracy: 0.7044 - val_loss: 1.1840 - val_accuracy: 0.6602 - lr: 3.7500e-05
Epoch 52/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1092 - accuracy: 0.7056  
Epoch 52: val_loss did not improve from 1.16846
1345/1345 [==============================] - 361s 268ms/step - loss: 1.1092 - accuracy: 0.7056 - val_loss: 1.1774 - val_accuracy: 0.6652 - lr: 3.7500e-05
Epoch 53/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1146 - accuracy: 0.7008  
Epoch 53: val_loss did not improve from 1.16846

Epoch 53: ReduceLROnPlateau reducing learning rate to 1.8750000890577212e-05.
1345/1345 [==============================] - 362s 269ms/step - loss: 1.1146 - accuracy: 0.7008 - val_loss: 1.1747 - val_accuracy: 0.6636 - lr: 3.7500e-05       
Epoch 54/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1019 - accuracy: 0.7055   
Epoch 54: val_loss did not improve from 1.16846
1345/1345 [==============================] - 360s 268ms/step - loss: 1.1019 - accuracy: 0.7055 - val_loss: 1.1703 - val_accuracy: 0.6688 - lr: 1.8750e-05
Epoch 55/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1042 - accuracy: 0.7040  
Epoch 55: val_loss did not improve from 1.16846
1345/1345 [==============================] - 362s 269ms/step - loss: 1.1042 - accuracy: 0.7040 - val_loss: 1.1700 - val_accuracy: 0.6652 - lr: 1.8750e-05
Epoch 56/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1031 - accuracy: 0.7059  
Epoch 56: val_loss did not improve from 1.16846

Epoch 56: ReduceLROnPlateau reducing learning rate to 9.375000445288606e-06.
1345/1345 [==============================] - 359s 267ms/step - loss: 1.1031 - accuracy: 0.7059 - val_loss: 1.1724 - val_accuracy: 0.6646 - lr: 1.8750e-05       
Epoch 57/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1002 - accuracy: 0.7080  
Epoch 57: val_loss improved from 1.16846 to 1.16602, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 380s 283ms/step - loss: 1.1002 - accuracy: 0.7080 - val_loss: 1.1660 - val_accuracy: 0.6643 - lr: 9.3750e-06
Epoch 58/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1003 - accuracy: 0.7088  
Epoch 58: val_loss improved from 1.16602 to 1.16557, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 380s 282ms/step - loss: 1.1003 - accuracy: 0.7088 - val_loss: 1.1656 - val_accuracy: 0.6650 - lr: 9.3750e-06
Epoch 59/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1048 - accuracy: 0.7076  
Epoch 59: val_loss improved from 1.16557 to 1.16409, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 361s 268ms/step - loss: 1.1048 - accuracy: 0.7076 - val_loss: 1.1641 - val_accuracy: 0.6621 - lr: 9.3750e-06
Epoch 60/60
1345/1345 [==============================] - ETA: 0s - loss: 1.1029 - accuracy: 0.7051  
Epoch 60: val_loss improved from 1.16409 to 1.15928, saving model to C:\XLAS\DoAn\XLAS_Project\Emotion_little_vgg.h5
1345/1345 [==============================] - 355s 264ms/step - loss: 1.1029 - accuracy: 0.7051 - val_loss: 1.1593 - val_accuracy: 0.6650 - lr: 9.3750e-06
"""

pattern = r"loss:\s([0-9.]+)\s-\saccuracy:\s([0-9.]+)\s-\sval_loss:\s([0-9.]+)\s-\sval_accuracy:\s([0-9.]+)"
matches = re.findall(pattern, log_text)

loss = []
acc = []
val_loss = []
val_acc = []

for m in matches:
    l, a, vl, va = m
    loss.append(float(l))
    acc.append(float(a))
    val_loss.append(float(vl))
    val_acc.append(float(va))

epochs = range(1, len(loss) + 1)

# =====================================================
# 3) VẼ TRAIN / VAL LOSS
# =====================================================

plt.figure(figsize=(10,5))
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================================================
# 4) VẼ TRAIN / VAL ACC
# =====================================================

plt.figure(figsize=(10,5))
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================================================
# 5) VẼ BIỂU ĐỒ TEST ACCURACY
# =====================================================
# 👉 Sau khi bạn evaluate:
# test_loss, test_acc = model.evaluate(test_gen)
# print(test_acc)
# → Ghi kết quả vào list dưới đây

test_acc_list = [0.6776214838027954]   # 👈 Thay bằng test acc thật của bạn

plt.figure(figsize=(6,4))
plt.plot(range(1, len(test_acc_list) + 1), test_acc_list, marker='o')
plt.title("Test Accuracy")
plt.xlabel("Test Run")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()