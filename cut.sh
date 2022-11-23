for subset in `seq 0 9`
do
python -W ignore infinite_generator_3D.py \
--fold $subset \
--scale 32 \
--data /dataset/nii_subset \
--save /dataset/generated_cubes
done


#--fold 0 --scale 32 --data /dataset/nii_subset --save /ckpt/generated_cubes