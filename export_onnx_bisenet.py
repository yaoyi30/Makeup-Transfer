# 此脚本放到 https://github.com/zllrunning/face-parsing.PyTorch 工程目录下

from models.bisenet import BiSeNet
import torch

x = torch.randn(1, 3, 512,512)
input_names = ["input"]
out_names = ["output"]

class FaceParsing(torch.nn.Module):
    def __init__(self,num_classes):
        super(FaceParsing,self).__init__()

        self.model = BiSeNet(num_classes)
        self.model.load_state_dict(torch.load('./weights/79999_iter.pth'))
        self.model.eval()

    def forward(self,x):
        x = self.model(x)[0]

        return x
if __name__ == '__main__':

    model = FaceParsing(19)

    print(model)

    torch.onnx.export(model, x, './face.onnx', export_params=True, training=False, input_names=input_names,
                      output_names=out_names, opset_version=13)
    print('please run: python -m onnxsim test.onnx test_sim.onnx\n')
