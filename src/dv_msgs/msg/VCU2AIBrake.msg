float32 hyd_press_f_pct       # Front hydraulic pressure percentage (0-100%)
float32 hyd_press_f_req_pct   # Front hydraulic pressure requirement percentage (0-100%)
float32 hyd_press_r_pct       # Rear hydraulic pressure percentage (0-100%)
float32 hyd_press_r_req_pct   # Rear hydraulic pressure requirement percentage (0-100%)
uint8 status_brk            # Brake system status (0-15)
uint8 status_ebs            # EBS system status (0-15)

# BO_ 1316 VCU2AI_Brake: 5 VCU
#  SG_ HYD_PRESS_F_pct : 0|8@1+ (0.5,0) [0|100] "%" Vector__XXX
#  SG_ HYD_PRESS_F_REQ_pct : 8|8@1+ (0.5,0) [0|100] "%" Vector__XXX
#  SG_ HYD_PRESS_R_pct : 16|8@1+ (0.5,0) [0|100] "%" Vector__XXX
#  SG_ HYD_PRESS_R_REQ_pct : 24|8@1+ (0.5,0) [0|100] "%" Vector__XXX
#  SG_ STATUS_BRK : 32|4@1+ (1,0) [0|15] "" Vector__XXX
#  SG_ STATUS_EBS : 36|4@1+ (1,0) [0|15] "" Vector__XXX