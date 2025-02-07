import pickle

our_case = pickle.load(open('decodermemoryl.pkl', 'rb'))
other_case = pickle.load(open('infer_seggpt_results.pkl', 'rb'))
other_case2 = pickle.load(open('infer_pfenet_results.pkl', 'rb'))
other_case3 = pickle.load(open('infer_vrpsam_results.pkl', 'rb'))

for i in range(len(our_case['pred_name'])):
    our_J = our_case['J_buf'][i].item()
    our_F = our_case['F_buf'][i].item()
    our_J_F = (our_case['J_buf'][i].item() + our_case['F_buf'][i].item()) / 2
    other_J = other_case['J_buf'][i].item()
    other_F = other_case['F_buf'][i].item()
    other_J_F = (other_case['J_buf'][i].item() + other_case['F_buf'][i].item()) / 2
    other_2_J = other_case2['J_buf'][i].item()
    other_2_F = other_case2['F_buf'][i].item()
    other_2_J_F = (other_case2['J_buf'][i].item() + other_case2['F_buf'][i].item()) / 2
    other_3_J = other_case3['J_buf'][i].item()
    other_3_F = other_case3['F_buf'][i].item()
    other_3_J_F = (other_case3['J_buf'][i].item() + other_case3['F_buf'][i].item()) / 2
    if our_J_F < 70.0 and our_J_F > 30.0:
        print(f'bad case: {our_case["pred_name"][i]}: J(our other): {our_J} {other_J}, F(our other): {our_F} {other_F}')
    # if our_J_F < (other_J_F - 20.0) and our_J_F < (other_2_J_F - 20.0) and our_J_F < (other_3_J_F - 20.0):
    #     print(f'bad case: {our_case["pred_name"][i]}: J(our other): {our_J} {other_J}, F(our other): {our_F} {other_F}')
    # if our_J_F > (other_J_F + 20.0) and our_J_F > (other_2_J_F + 20.0) and our_J_F > (other_3_J_F + 20.0):
    #     print(f'good case: {our_case["pred_name"][i]}: J(our other): {our_J} {other_J} {other_2_J}, F(our other): {our_F} {other_F} {other_2_F}')
    