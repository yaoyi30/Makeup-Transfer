# 此脚本放到 https://github.com/Snowfallingplum/CSD-MT 工程目录下

from CSD_MT.modules import Generator
import torch

class MakeUp(torch.nn.Module):
    def __init__(self):
        super(MakeUp,self).__init__()

        self.model = Generator(input_dim=3,parse_dim=10,ngf=16,device='cpu')
        checkpoint = torch.load('./CSD_MT/weights/CSD_MT.pth',map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['gen'])
        self.model.eval()

    def forward(self,x):

        source_img, source_parse, source_all_mask, ref_img, ref_parse, ref_all_mask = x[0],x[1],x[2],x[3],x[4],x[5]
        output = self.model(source_img=source_img,
                                             source_parse=source_parse,
                                             source_all_mask=source_all_mask,
                                             ref_img=ref_img,
                                             ref_parse=ref_parse,
                                             ref_all_mask=ref_all_mask)

        return output


if __name__ == '__main__':

    input_names = ['input1', 'input2', 'input3','input4', 'input5', 'input6']
    output_names = ['output']

    source_img = torch.randn(1, 3, 512, 512)
    source_parse = torch.randn(1,10, 512, 512)
    source_all_mask = torch.randn(1, 3, 512, 512)
    ref_img = torch.randn(1, 3, 512, 512)
    ref_parse = torch.randn(1, 10, 512, 512)
    ref_all_mask = torch.randn(1, 3, 512, 512)

    model_input = [source_img,source_parse,source_all_mask,ref_img,ref_parse,ref_all_mask]

    model = MakeUp()

    print(model)

    torch.onnx.export(model, model_input, './makeup.onnx', export_params=True, training=False, input_names=input_names,
                      output_names=output_names, opset_version=13)
    print('please run: python -m onnxsim test.onnx test_sim.onnx\n')