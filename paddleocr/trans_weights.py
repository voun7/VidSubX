import paddle
import torch
import numpy as np


def trans():
    input_fp = "./demo.pth.tar"
    output_fp = "./aster_torch.pdparams"
    #torch_dict = torch.load(input_fp)["model"]
    torch_dict = torch.load(input_fp, map_location="cpu")
    state_dict = torch_dict["state_dict"]
    paddle_dict = {}
    fc_names = ["qkv.weight", "fc1.weight", "fc2.weight", "downsample.reduction.weight", "head.weight", "attn.proj.weight"]
    fc_names = []
    for key in state_dict:
        weight = state_dict[key].cpu().numpy()
        flag = [i in key for i in fc_names]
        flag = "fc" in key or "wEmbed.weight" in key
        if flag:
            #print("weight {} need to be trans".format(key))
            weight = weight.transpose()
        if "stn_head" in key:
            key = "transform."+key
        if "encoder" in key:
            key = key.replace("encoder","backbone")
        if "running_mean" in key:
            key = key.replace("running_mean", "_mean")
        if "running_var" in key:
            key = key.replace("running_var", "_variance")
        if "decoder.decoder" in key:
            key = key.replace("decoder.decoder", "head.decoder")
        if "gru" in key:
            key = key[:-3]
        if "num_batches_tracked" in key:
            continue
        print(key)
        paddle_dict[key] = weight

    paddle.save(paddle_dict, output_fp)
    print("ok")
    return

def trans_sar():
    input_fp = "./sar_r31_parallel_decoder_academic-dba3a4a3.pth"
    output_fp = "./sar_torch_fix.pdparams"
    #torch_dict = torch.load(input_fp)["model"]
    torch_dict = torch.load(input_fp, map_location="cpu")
    state_dict = torch_dict["state_dict"]
    # for key in state_dict:
    #     print(key, state_dict[key].shape)
    # import pdb
    # pdb.set_trace()
    paddle_dict = {}
    fc_names = ["encoder.linear.weight", "decoder.prediction.weight", "decoder.conv1x1_1.weight", "decoder.conv1x1_2.weight"]
    for key in state_dict:
        weight = state_dict[key].cpu().numpy()
        flag = key in fc_names
        if flag:
            print('ori_shape', weight.shape)
            weight = weight.transpose()
            print('key', key)
            print('shape',weight.shape)
        if 'encoder' in key:
            if "encoder.rnn_encoder" in key:
                key = key.replace("encoder.rnn_encoder","head.encoder.rnn_encoder")
            else:
                key = key.replace("encoder", "head.encoder")
        if "running_mean" in key:
            key = key.replace("running_mean", "_mean")
        if "running_var" in key:
            key = key.replace("running_var", "_variance")
        if "decoder" in key:
            if "decoder.rnn_decoder" in key:
                key = key.replace("decoder.rnn_decoder", "head.decoder.rnn_decoder")
            else:
                key = key.replace("decoder", "head.decoder")
        if "num_batches_tracked" in key:
            continue
        # print(key)
        paddle_dict[key] = weight

    paddle.save(paddle_dict, output_fp)
    print("ok")
    return

def trans_sar_ch():
    input_fp = "./sar_r31_parallel_decoder_chineseocr_20210507-b4be8214.pth"
    output_fp = "./sar_torch_chinese.pdparams"
    #torch_dict = torch.load(input_fp)["model"]
    torch_dict = torch.load(input_fp, map_location="cpu")
    state_dict = torch_dict["state_dict"]
    for key in state_dict:
        print(key, state_dict[key].shape)
    import pdb
    pdb.set_trace()
    paddle_dict = {}
    fc_names = ["encoder.linear.weight", "decoder.prediction.weight", "decoder.conv1x1_1.weight", "decoder.conv1x1_2.weight"]
    for key in state_dict:
        weight = state_dict[key].cpu().numpy()
        flag = key in fc_names
        if flag:
            print('ori_shape', weight.shape)
            weight = weight.transpose()
            print('key', key)
            print('shape',weight.shape)
        if 'encoder' in key:
            if "encoder.rnn_encoder" in key:
                key = key.replace("encoder.rnn_encoder","head.encoder.rnn_encoder")
            else:
                key = key.replace("encoder", "head.encoder")
        if "running_mean" in key:
            key = key.replace("running_mean", "_mean")
        if "running_var" in key:
            key = key.replace("running_var", "_variance")
        if "decoder" in key:
            if "decoder.rnn_decoder" in key:
                key = key.replace("decoder.rnn_decoder", "head.decoder.rnn_decoder")
            else:
                key = key.replace("decoder", "head.decoder")
        if "num_batches_tracked" in key:
            continue
        # print(key)
        paddle_dict[key] = weight

    paddle.save(paddle_dict, output_fp)
    print("ok")
    return

def trans_satrn():
    input_fp = "./satrn_academic_20211009-cb8b1580.pth"
    output_fp = "./satrn_torch.pdparams"
    torch_dict = torch.load(input_fp, map_location="cpu")
    state_dict = torch_dict["state_dict"]
    paddle_dict = {}
    
    fc_names = ['linear_q', 'linear_k', 'linear_v', 'fc', 'classifier', 'mlp']
    for key in state_dict:
        weight = state_dict[key].cpu().numpy()
        if key in fc_names:
            weight = weight.transpose()
        
        if 'backbone' in key:
            if 'conv1.conv' in key:
                key = key.replace("conv1.conv", "conv1")
            elif 'conv1.bn' in key:
                key = key.replace("conv1.bn", "bn1")
            elif 'conv2.conv' in key:
                key = key.replace("conv2.conv", "conv2")
            elif 'conv2.bn' in key:
                key = key.replace("conv2.bn", "bn2")
            else:
                print('{} is unknown'.format(key))
        elif 'encoder' in key:
            if 'feed_forward' in key:
                if 'conv1.conv' in key:
                    key = key.replace("conv1.conv", "conv1")
                elif 'conv1.bn' in key:
                    key = key.replace("conv1.bn", "bn1")
                elif 'depthwise_conv.conv' in key:
                    key = key.replace("depthwise_conv.conv", "depthwise_conv")
                elif 'depthwise_conv.bn' in key:
                    key = key.replace("depthwise_conv.bn", "bn2")
                elif 'conv2.conv' in key:
                    key = key.replace("conv2.conv", "conv2")
                elif 'conv2.bn' in key:
                    key = key.replace("conv2.bn", "bn3")
                else:
                    print('{} is unknown'.format(key))
        elif 'decoder' in key:
            if 'trg_word_emb' in key:
                key = key.replace('trg_word_emb', 'tgt_word_emb')

        if 'encoder.' in key:
            key = key.replace('encoder.', 'head.encoder.')
        if 'decoder.' in key:
            key = key.replace('decoder.', 'head.decoder.')
        if "running_mean" in key:
            key = key.replace("running_mean", "_mean")
        if "running_var" in key:
            key = key.replace("running_var", "_variance")
        if "num_batches_tracked" in key:
            continue
        paddle_dict[key] = weight

    paddle.save(paddle_dict, output_fp)
    print("ok")
    return

def trans_visionlan(input_fp, output_fp):
    paddle_model = VisionLAN()
    paddle_dict = paddle_model.state_dict()
    torch_dict = torch.load(input_fp, map_location="cpu")
    paddle_dict_keys = paddle_dict.keys()
    torch_dict_keys = torch_dict.keys()
    # check diff
    print('torch')
    right = 0
    wrong = 0
    for torch_key in torch_dict_keys:
        torch_key = torch_key.replace('module.', '')
        torch_key = torch_key.replace('running_mean', '_mean')
        torch_key = torch_key.replace("running_var", "_variance")
        if "num_batches_tracked" in torch_key:
            continue
        if torch_key not in paddle_dict_keys:
            print(torch_key)
            wrong += 1
        else:
            right += 1
    
    print('right : ', right)
    print('wrong : ', wrong)
    # check diff
    print('paddle')
    right = 0
    wrong = 0
    for padddle_key in paddle_dict_keys:
        padddle_key = 'module.' + padddle_key
        padddle_key = padddle_key.replace('_mean', 'running_mean')
        padddle_key = padddle_key.replace("_variance", "running_var")
        if "num_batches_tracked" in torch_key:
            continue
        if padddle_key not in torch_dict_keys:
            print(padddle_key)
            wrong += 1
        else:
            right += 1
    
    print('right : ', right)
    print('wrong : ', wrong)
    # end check diff

    # trans Linear shape
    fc_names = ['MLM.w0_linear', 'MLM.wv', 'MLM.we', 
                'fc', 'slf_attn.w_qs', 'slf_attn.w_ks', 'slf_attn.w_vs', 
                'pp.w0', 'pp.wv', 'pp.we', 
                'pp_share.w0', 'pp_share.wv', 'pp_share.we', 
                'w_vrm', 'w_share']
    for key in torch_dict_keys:
        weight = torch_dict[key].cpu().numpy()
        for fc in fc_names:
            if fc in key and 'bias' not in key:
                print('key: ', key)
                weight = weight.transpose()
        if "running_mean" in key:
            key = key.replace("running_mean", "_mean")
        if "running_var" in key:
            key = key.replace("running_var", "_variance")
        if "num_batches_tracked" in key:
            continue
        key = key.replace('module.', '')
        paddle_dict[key] = weight
    paddle.save(paddle_dict, output_fp)
    print("ok")
    return

if __name__ == "__main__":
    # input_fp = "./pretrained_model/LF_1.pth"
    output_fp = "./pretrained_model/LF_2.pdparams"
    # trans_visionlan(input_fp, output_fp)
    # torch_dict = torch.load(input_fp, map_location="cpu")['state_dict']
    paddle_dict = paddle.load(output_fp)
    paddle_dict_keys = paddle_dict.keys()
    ocr_dict = {}
    # import pdb
    # pdb.set_trace()
    for padddle_key in paddle_dict_keys:
        val = paddle_dict[padddle_key]
        if 'backbone' not in padddle_key:
            padddle_key = 'head.' + padddle_key
        ocr_dict[padddle_key] = val
    paddle.save(ocr_dict, "./pretrained_model/LF_2_ocr.pdparams")
    print()

    