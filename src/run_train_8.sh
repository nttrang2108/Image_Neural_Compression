python train_neural_compressor_8.py --entropy_coding_type="arm" \
                                    --D=64 \
                                    --C=16 \
                                    --E=8 \
                                    --M=256 \
                                    --M_kernels=32 \
                                    --batch_size=64 \
                                    --epochs=500 \
                                    --seed=42 \
                                    --lr=0.0001 \
                                    --max_patience=50
