def Loss_function(v_old, v_new, flow_mask):
    v = (v_new + v_old) / 2
    loss_mom = 