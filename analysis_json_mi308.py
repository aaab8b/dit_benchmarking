import json
import pandas as pd
import sys


def read_json(file_name):
    with open(file_name, 'r') as json_file:
        data = json.load(json_file)
        if 'traceEvents' in data:
            data = data['traceEvents']
    return data

def get_name_shape(data):
    out = [None, None]
    if ('name' in data) and ('args' in data):
        cur_op_name = data['name']
        cur_args = data['args']
        if 'Input Dims' in cur_args:
            cur_shape = cur_args['Input Dims']
            out = [cur_op_name, cur_shape]
    return out
def get_single_op(data, op_name, shape):
    out_num = 0
    op = get_name_shape(data)
    if op[0] is not None:
        if (op[0] == op_name) and (op[1] == shape):
            out_num = 1
    return out_num

def get_all_shape(data, op_name):
    out = None
    op = get_name_shape(data)
    if (op[0] == op_name):
        out = op[1]
    return out

def append_if_not_exists(lst, element):
    if element not in lst:
        lst.append(element)


def get_op_num(json_data, op_name, shape):
    evens = json_data
    out_nums = 0
    out_shapes = []
    for cur in evens:
        out_nums += get_single_op(cur, op_name, shape)
        out_shape = get_all_shape(cur, op_name)
        append_if_not_exists(out_shapes, out_shape)
    return out_nums, out_shapes

def get_single_kernel_id(data, kernel_name):
    out = [None, None, None]
    if ('name' in data) and ('args' in data) and ('dur' in data):
        if (kernel_name in data['name']) and ('External id' in data['args']):
            out = [data['args']['External id'], data['dur'], data['name']]
    return out

def kernel_name_map(kernels):
    out_map={}
    for cur in kernels:
        key = cur[0]
        val = cur[1]
        if key not in out_map:
            out_map[key] = val
        else:
            cur_val = out_map[key]
            cur_val += val
            out_map[key] = cur_val
    out_dict_sort = dict(sorted(out_map.items(), key=lambda item: item[1], reverse=True))
    return out_dict_sort

def get_single_kernel_name_time(data, kernel_name):
    out = (None, None)
    if ('name' in data)  and ('dur' in data) and ('cat' in data):
        if data['cat'] == 'kernel':
            if kernel_name in data['name']:
                out = (data['name'], data['dur'])
    return out

def get_all_name(data, kernel_name):
    evens = data
    all_names = []
    out_names = []
    for cur in evens:
        cur_name = get_single_kernel_name_time(cur, kernel_name)
        if cur_name[0] is not None:
            all_names.append(cur_name)
    names_dict = kernel_name_map(all_names)
    print(" sort by time")
    idx = 0
    for key in names_dict.keys():
        idx += 1
        cur_str = 'kernel_name' + str(idx) + ' = "{}"'.format(key)
        out_names.append(key)
        print(cur_str)
    return out_names


def list_to_tuple(obj):
    if isinstance(obj, list):
        return tuple(list_to_tuple(item) for item in obj)
    else:
        return obj


def deal_all_kernel2(kernels):
    out_dict = {}
    for cur in kernels:
        name = cur['name']
        shape = list_to_tuple(cur['shape'])
        type = list_to_tuple(cur['type'])
        strid = list_to_tuple(cur['stride'])
        key = (name, shape, type, strid)
        dur = cur['kernel_dur']
        # op_ts = cur['op_ts']
        # op_dur = cur['op_dur']
        attach = cur['attach']
        if key not in out_dict:
            kernel_dur = [1, dur, dur, dur]
            value = {'kernel_dur': kernel_dur, 'attach': attach}
            out_dict[key] = value
        else:
            value_src = out_dict[key]
            value = value_src['kernel_dur']
            value[0] += 1
            value[1] += dur
            value[2] = max(value[2], dur)
            value[3] = min(value[3], dur)
            value_src['kernel_dur'] = value
            out_dict[key] = value_src
    out_dict_sort = dict(sorted(out_dict.items(), key=lambda item: item[1]['kernel_dur'][1],reverse=True))
    return out_dict_sort


def find_same_id2(event, obj_id, search_range, last_loc):
    out = None
    lens = len(event)
    for idx in range(last_loc, lens, 1):
        cur_data = event[idx]
        if ('name' in cur_data) and ('args' in cur_data):
            if 'External id' in cur_data['args']:
                cur_args = cur_data['args']
                cur_id = cur_args['External id']
                cur_name = cur_data['name']
                if (cur_id == obj_id):
                    if cur_name in search_range:
                        cur_shape = cur_args['Input Dims']
                        cur_type = cur_args['Input type']
                        cur_strid = cur_args['Input Strides']
                        attach = {'op_ts':cur_data['ts'], 'op_dur':cur_data['dur'], 'pid':cur_data['pid'],'tid':cur_data['tid'] }
                        out = {'name':cur_name, 'shape': cur_shape, 'type': cur_type, 'stride': cur_strid, 'idx': idx, 'attach':attach}
                        break
    return out

def look_k(arr):
    my_dict = {}
    out = None
    for cur in arr:
        if cur not in my_dict.keys():
            my_dict[cur] = 1
        else:
            num = my_dict[cur]
            num += 1
            my_dict[cur] = num
    for key in my_dict.keys():
        num = my_dict[key]
        if num >= 2:
            out = key
            break
    return out


def deal_shape(name, kernel_shape):
    # if name in ('aten::mm', 'aten::addmm'):
    if name in ('aten::mm'):
        M = kernel_shape[0][0]
        N = kernel_shape[1][1]
        K = kernel_shape[0][1]
        kernel_shape = {'src': kernel_shape,'MNK': (M, N, K)}
    if name in ('aten::addmm'):
        M = kernel_shape[1][0]
        N = kernel_shape[2][1]
        K = kernel_shape[1][1]
        kernel_shape = {'src': kernel_shape,'MNK': (M, N, K)}
    if name in ('GroupedGemm', 'GroupedGemm'):
        M = kernel_shape[0][0]
        N = kernel_shape[1][2]
        K = kernel_shape[0][1]
        batch = kernel_shape[1][0]
        # N = batch * N
        kernel_shape = {'src': kernel_shape, 'MNK': (M, N, K)}
    if name in ('tex_ts::te_gemm_ts','tex_ts::te_gemm_ts'):
        M = kernel_shape[10][0]
        N = kernel_shape[10][1]
        # K = kernel_shape[0][0]
        K = look_k([kernel_shape[0][0], kernel_shape[0][1], kernel_shape[5][0], kernel_shape[5][1]])
        kernel_shape = {'src': kernel_shape, 'MNK': (M, N, K)}
    return kernel_shape


def insert_mnk2(data, info):
    data['kernel_name'].append(info['name'])
    data['M'].append(info['M'])
    data['N'].append(info['N'])
    data['K'].append(info['K'])
    data['kernel_num'].append(info['kernel_num'])
    data['time_sum'].append(info['time_sum'])
    data['time_max'].append(info['time_max'])
    data['time_min'].append(info['time_min'])
    data['dtype_info'].append(info['dtype_info'])
    data['kernel_idx'].append(info['idx'])
    return data
def write_to_excel(out_file, kernel_info, skip_time=5.0):

    idx = int(0)
    out_data = {'kernel_idx': [], 'kernel_name': [], 'M':[], 'N':[], 'K':[],'kernel_num': [],'time_sum':[], 'time_max':[],'time_min':[], 'dtype_info':[]}
    for cur_kernel_info in kernel_info:
        idx += 1
        kernel_idx = 'kernel_name' + str(idx)
        for key in cur_kernel_info.keys():
            name = key[0]
            shape_info = key[1]
            shape_info = deal_shape(name, shape_info)
            value_time = cur_kernel_info[key]['kernel_dur']
            time_sum = value_time[1]
            if time_sum < skip_time:
                continue  #æ€»è€—æ—¶å°äºŽ1msçš„å°±pass
            dtype_info = key[2]
            stride_info = key[3]
            dtype_info = {'name':name,'shape': shape_info, 'dtype':dtype_info, 'stride_info':stride_info }
            if 'parent_attach' in cur_kernel_info[key]:
                parent_attach = cur_kernel_info[key]['parent_attach']
                dtype_info = (parent_attach, dtype_info)
            if name in ('aten::mm', 'aten::addmm', 'GroupedGemm', 'tex_ts::te_gemm_ts'):
                shape_info = shape_info['MNK']
                info_d = {'name':name, 'M': shape_info[0], 'N': shape_info[1], 'K': shape_info[2], 'kernel_num': value_time[0],'time_sum': value_time[1], 'time_max':value_time[2], \
                          'time_min': value_time[3], 'dtype_info': dtype_info, 'idx': kernel_idx}
                out_data = insert_mnk2(out_data, info_d)
            else:
                info_d = {'name': name, 'M': '', 'N': '', 'K': '',
                          'kernel_num': value_time[0], 'time_sum': value_time[1], 'time_max': value_time[2], \
                          'time_min': value_time[3], 'dtype_info': dtype_info, 'idx': kernel_idx}
                # out_data = insert_mnk2(out_data, [name, '', '', '', value_time[0], value_time[1], value_time[2], value_time[3], dtype_info, kernel_idx])
                out_data = insert_mnk2(out_data, info_d)
        info3 = {'name': '', 'M': '', 'N': '', 'K': '', 'kernel_num': '', 'time_sum': '', 'time_max': '', 'time_min': '', 'dtype_info': '', 'idx': ''}
        # å†™å…¥ç©ºè¡Œ
        # out_data = insert_mnk2(out_data, info3)

    df = pd.DataFrame(out_data)
    df.to_excel(out_file, sheet_name='Sheet1', index=False)
def print_kernels(kernels, skip_time=5.0):
    print("opName, kernelNum, allTime(ms), maxTime(ms), shape, type, strid")
    for key in kernels.keys():
        value = kernels[key]['kernel_dur']
        value[1] = value[1] / 1000
        value[2] = value[2] / 1000
        value[3] = value[3] / 1000
        dtype_info = key[2]
        shape_info = key[1]
        stride_info = key[3]
        name = key[0]

        if value[1] < skip_time:
            continue
        # dtype_info = shape_info + dtype_info + stride_info
        dtype_info = {'name': name,'shape:': shape_info, 'dtype':dtype_info, 'stride_info': stride_info}
        if 'parent_attach' in kernels[key]:
            parent_attach = kernels[key]['parent_attach']
            dtype_info = (parent_attach, dtype_info)

        shape_info = deal_shape(name, shape_info)
        print(name, " | ",value, " | ", shape_info, " | ", dtype_info)
    print(' ')


def get_category_op(data, category='cpu_op'):
    cpu_ops = []
    for cur_dict in data:
        if ('name' in cur_dict) and ('cat' in cur_dict):
            if cur_dict['cat'] == category:
                cpu_ops.append(cur_dict)
    return cpu_ops

def get_cpu_op(op_data, op_name):
    dst_ops = []
    for cur_dict in op_data:
        if op_name == cur_dict['name']:   #string search type
            dst_ops.append(cur_dict)
    return dst_ops

def search_parament(cpu_json, pid, tid, op_ts, op_dur):
    out = None
    sim_op = []
    idx = 0
    for cur in cpu_json:
        if (cur['pid'] == pid) and (cur['tid'] == tid):
            para_ts = cur['ts']
            para_end = para_ts + cur['dur']
            child_ts = op_ts
            child_end = child_ts + op_dur
            if (child_ts > para_ts) and (child_end < para_end) and (cur['dur'] > op_dur):
                sim_op.append(cur)
                idx += 1

            if idx > 0:
                idx += 1
            # if idx > 400:
            #     break
    max_ts = -1.0  #sys.float_info.max
    act = None
    for op in sim_op:
        if op['ts'] > max_ts:
            max_ts = op['ts']
            act = op
    out = act
    return out

def mm_parament(cpu_json, kernels_info):
    for key in kernels_info.keys():
        if key[0] in ('aten::mm', 'aten::mm'):
            value = kernels_info[key]
            attach = value['attach']
            op_ts = attach['op_ts']
            op_dur = attach['op_dur']
            pid = attach['pid']
            tid = attach['tid']
            para_op = search_parament(cpu_json, pid, tid, op_ts, op_dur)
            para_args = para_op['args']
            para_attach = {'parent_name':para_op['name'], 'Input Dims':para_args['Input Dims'], 'Strides': para_args['Input Strides'], 'type': para_args['Input type']}
            value['parent_attach'] = para_attach
            kernels_info[key] = value
    return kernels_info


'Input type''Input Strides''Input Dims'  'args''name'

def get_kernel_shape_time(json_data, kernel_name, search_range):
    evens = json_data
    cpu_json = get_category_op(evens)
    external_ids = []
    for cur in evens:
        id = get_single_kernel_id(cur, kernel_name)
        if id[0] is not None:
            external_ids.append(id)

    kernels_info = []
    idx = 0
    last_loc = 0
    for cur_id in external_ids:
        obj_id = cur_id[0]
        out = find_same_id2(cpu_json, obj_id, search_range, last_loc)
        dur = cur_id[1]
        # last_loc = loc
        out['kernel_dur'] = dur
        kernels_info.append(out)
        idx += 1
        if idx % 100 == 0:
            print(idx, obj_id)
    kernels_sum = deal_all_kernel2(kernels_info)
    kernels_sum = mm_parament(cpu_json, kernels_sum)
    return kernels_sum

def find_kernel_id_by_cpu(kernel_op, id):
    out = []
    idx = int(0)
    for cur_kernel in kernel_op:
        kernel_id = cur_kernel['args']['External id']
        if kernel_id == id:
            out.append(cur_kernel)
            idx += 1
        if idx > 0:
            idx += 1
        # if idx > 80:
        #     break
    return out

def cal_kernel_time(input):
    out = []
    for cur in input:
        lens = len(cur)
        dur = 0.0
        if lens == 1:
            dur = cur[0]['dur']
        if lens > 1:
            min_start = cur[0]['ts']
            max_end = min_start + cur[0]['dur']
            for cur_kernel in cur:
                start = cur_kernel['ts']
                end = start + cur_kernel['dur']
                min_start = min(min_start, start)
                max_end = max(max_end, end)
            dur = max_end - min_start
        out.append(dur)
    return out


def reduce_kernel(cpu_op, kernel_time, skip_time_ms=5.0):
    lens = len(cpu_op)
    assert lens == len(kernel_time)
    op_dict = {}
    for idx in range(lens):
        cur_cpu = cpu_op[idx]
        name = cur_cpu['name']
        args = cur_cpu['args']
        shape = args['Input Dims']
        types = args['Input type']
        stride = None
        if 'Input Strides' in args:
            stride = args['Input Strides']
        key = (name, list_to_tuple(shape),list_to_tuple(types),list_to_tuple(stride))    #list_to_tuple
        dur = kernel_time[idx]
        if key not in op_dict.keys():
            times = [1, dur, dur, dur]  #count, sum_time, max_time, min_time
            op_dict[key] = {'attach': cur_cpu, 'kernel_dur':times}
        else:
            value = op_dict[key]
            times = value['kernel_dur']
            times[0] += 1
            times[1] += dur
            times[2] = max(dur, times[2])
            times[3] = min(dur, times[3])
            value['kernel_dur'] = times
            op_dict[key] = value
    op_dict_del = {}
    for key in op_dict.keys():
        time = op_dict[key]['kernel_dur'][1]
        if time >= skip_time_ms * 1000:
            op_dict_del[key] = op_dict[key]
    out_dict_sort = dict(sorted(op_dict_del.items(), key=lambda item: item[1]['kernel_dur'][1], reverse=True))
    return out_dict_sort






def get_op_shape_times(data, kernel_name,skip_time_ms=5.0):
    cpu_json = get_category_op(data)
    obj_op = get_cpu_op(cpu_json, kernel_name)
    kernel_op = get_category_op(data, 'kernel')
    obj_para = []
    obj_kernel = []
    idx = int(0)
    for cur_obj in obj_op:
        id = cur_obj['args']['External id']
        kernel = find_kernel_id_by_cpu(kernel_op, id)
        # para = search_parament(cpu_json, cur_obj['pid'], cur_obj['tid'], cur_obj['ts'], cur_obj['dur'])
        obj_kernel.append(kernel)

        # obj_para.append(para)
    kernel_times = cal_kernel_time(obj_kernel)
    sort_op = reduce_kernel(obj_op, kernel_times, skip_time_ms)
    for key in sort_op.keys():
        idx += 1
        if idx % 50 == 0:
            print(idx)
        value = sort_op[key]
        cur_obj = value['attach']
        para = search_parament(cpu_json, cur_obj['pid'], cur_obj['tid'], cur_obj['ts'], cur_obj['dur'])
        p_args = para['args']
        parent_info = {'name': para['name'], 'shape':p_args['Input Dims'], 'stride':p_args['Input Strides'], 'type': p_args['Input type']}
        p_args = cur_obj['args']
        op_info = {'name': cur_obj['name'], 'shape': p_args['Input Dims'], 'stride': p_args['Input Strides'], 'type': p_args['Input type']}
        value['parent_attach'] = parent_info
        value['attach'] = op_info
        sort_op[key] = value
    return sort_op

def get_all_cpu_op_times(data, cpu_names, excel_file, skip_time=5.0):
    infos = []
    for name in cpu_names:
        info = get_op_shape_times(data, name, skip_time)
        infos.append(info)
    for idx in range(len(infos)):
        print("kernel name", idx + 1)
        print("kernel name:", cpu_names[idx])
        print_kernels(infos[idx], skip_time)
    write_to_excel(excel_file, infos, skip_time)


def get_kernel_shape_times(data, kernel_names, search_range, excel_file, skip_time=5.0):
    infos = []
    for name in kernel_names:
        print("kernel name:{}".format(name))
        info = get_kernel_shape_time(data, name, search_range)
        infos.append(info)
    for idx in range(len(infos)):
        print("kernel name", idx + 1)
        print("kernel name:", kernel_names[idx])
        # print("info")
        # print(infos[idx].keys())
        print_kernels(infos[idx], skip_time)
    write_to_excel(excel_file, infos, skip_time)




search_range = ['aten::mm','aten::addmm', 'tex_ts::te_gemm_ts', 'GroupedGemm', 'GroupedGemmBackwards', 'GroupedGemmBackward']
search_range += ['AttnFuncWithCP', 'AttnFuncWithCPBackward', 'lash_attn::_flash_attn_forward', 'GeneratedBackwardFor_flash_attn__flash_attn_forward_default', 'GeneratedBackwardFor_flash_attn__flash_attn_backward_default']
search_range += ['_LayerNormLinearBackward', '_LayerNormLinear']
search_range += ['triton_poi_fused__to_copy_add_mul_0', 'triton_poi_fused_cat_0','triton_poi_fused_mul_silu_0','triton_poi_fused_add_0','triton_poi_fused_add_copy_exp_log_maximum_minimum_sub_0']
search_range += ['aten::add_', 'aten::mul','aten::copy_', 'aten::neg','aten::fill_','aten::add','aten::silu', 'aten::index','aten::mul_', 'aten::scatter_add_', 'aten::gather', 'aten::silu_backward']
search_range += ["aten::convolution_backward","aten::cudnn_convolution"]

# file_path = "/Users/hanzhihua/Desktop/profiler/Qwen1.5_14B_mi308_bs32_trace_1203.json"
# file_path = "/Users/hanzhihua/Desktop/profiler/megatron_mi308_1203.json"

# file_path = "/Users/hanzhihua/Desktop/profiler/Qwen1.5_14B_mi308_1206.json"
# file_path = "/Users/hanzhihua/Desktop/profiler/qwen2-57B-mi308-1206.json"


# file_path = "/Users/hanzhihua/Desktop/profiler/megtran_cp_mi308_1206.json"
# file_path = "/Users/hanzhihua/Desktop/profiler/moe-18b_mi308_1206.json"
# file_path = "/Users/hanzhihua/Desktop/profiler/QWen2-57b-mi308-1213.json"
# file_path = "/Users/hanzhihua/Desktop/profiler/megatron-cp_mi308_1216.json"


# file_path = "/Users/hanzhihua/Desktop/profiler/torch_profile_group_gemm11111.json"
# file_path = "/Users/hanzhihua/Desktop/profiler/moe-18b_h20_1217.json"
file_path="/root/data/trace_use_hipblast.json"


out_excel_file = '/root/data/MI308_model_kernel_ratio.xlsx'  #xlsx
out_op_excel = '/root/data/MI308_cpu_op_ratio.xlsx'  #xlsx
skip_time_ms = 3.0   #æ€»è€—æ—¶å°äºŽ3mså°†ä¼špass

data = read_json(file_path)

#é€‚ç”¨äºŽgroup gemm
get_all_cpu_op_times(data, ('GroupedGemm',), out_op_excel, skip_time_ms)

# names = get_all_name(data, "Fmha")
# names = get_all_name(data, "kernel_func")

names = get_all_name(data, "Cijk")
#names = get_all_name(data, "gemm")
# names = get_all_name(data, "rmsnorm")
# names = get_all_name(data, "triton_")
# names = get_all_name(data, "elementwise")

#names = names[:20]  #é€‰å–å‰å‡ ä¸ªè€—æ—¶æ¯”è¾ƒå¤§çš„
#æ ¹æ®kernelçš„è€—æ—¶æŽ’åºï¼Œè®¡ç®—opè€—æ—¶
#é€‚ç”¨äºŽéžgroup gemm
info = get_kernel_shape_times(data, names, tuple(search_range), out_excel_file, skip_time_ms)

