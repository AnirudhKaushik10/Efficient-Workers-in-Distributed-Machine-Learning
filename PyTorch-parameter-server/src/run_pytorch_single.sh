python -m torch.distributed.launch \
--nproc_per_node=3 \
/content/drive/MyDrive/DISML/Project/PyTorch-parameter-server/src/distributed_nn.py \
--lr=0.01 \
--momentum=0.9 \
--max-steps=10000 \
--epochs=10 \
--network=LeNet \
--dataset=MNIST \
--batch-size=64 \
--comm-type=Bcast \
--num-aggregate=5 \
--mode=normal \
--eval-freq=20 \
--gather-type=gather \
--compress-grad=compress \
--enable-gpu= \
--train-dir=/content/drive/MyDrive/DISML/Project/PyTorch-parameter-server/src/