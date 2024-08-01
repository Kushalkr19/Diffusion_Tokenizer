#rm -rf data/
#rm -rf models/pretrained
#mkdir data/
#mkdir models/pretrained
fusermount -u data/
fusermount -u models/pretrained/
gcsfuse --implicit-dirs fdl-2024-lunar-global-map "data/"
gcsfuse --implicit-dirs fdl_model_weights "models/pretrained/"

#fusermount -u data/


