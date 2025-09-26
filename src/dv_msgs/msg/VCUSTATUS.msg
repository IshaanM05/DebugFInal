uint8 sm_sys                            # System mode status (0-15)
uint8 sm_as                             # AS system mode status (0-15)
bool r1_ai2vcu_status_timeout_error     # Timeout error flag for AI2VCU status (0-1)
uint8 sys_action_state                  # System action state (0-3)
bool warn_ai_estop_req                  # AI emergency stop request warning (0-1)
bool warn_auto_braking                  # Auto braking warning (0-1)
bool warn_mission_status                # Mission status warning (0-1)
bool r1_ai2vcu_drive_f_timeout_error    # Timeout error for AI2VCU front drive (0-1)
bool r1_ai2vcu_drive_r_timeout_error    # Timeout error for AI2VCU rear drive (0-1)
bool r1_ai2vcu_steer_timeout_error      # Timeout error for AI2VCU steering (0-1)
bool r1_ai2vcu_brake_timeout_error      # Timeout error for AI2VCU braking (0-1)
bool r1_ai2vcu_status_handshake_timeout_error # Timeout error for AI2VCU status handshake (0-1)
bool warn_kl15_under_v                  # KL15 under voltage warning (0-1)
bool warn_brake_plausibility            # Brake plausibility warning (0-1)
bool warn_ai_comms_lost                 # AI communication lost warning (0-1)

# BO_ 288 VCU_STATUS: 8 VCU
#  SG_ SM_SYS : 0|4@1+ (1,0) [0|15] "" Vector__XXX
#  SG_ SM_AS : 12|4@1+ (1,0) [0|15] "" Vector__XXX
#  SG_ R1_AI2VCU_STATUS_TIMEOUT_ERROR : 32|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ SYS_ACTION_STATE : 48|4@1+ (1,0) [0|3] "" Vector__XXX
#  SG_ WARN_AI_ESTOP_REQ : 60|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ WARN_AUTO_BRAKING : 62|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ WARN_MISSION_STATUS : 63|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ R1_AI2VCU_DRIVE_F_TIMEOUT_ERROR : 38|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ R1_AI2VCU_DRIVE_R_TIMEOUT_ERROR : 39|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ R1_AI2VCU_STEER_TIMEOUT_ERROR : 46|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ R1_AI2VCU_BRAKE_TIMEOUT_ERROR : 47|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ R1_AI2VCU_STATUS_HANDSHAKE_TIMEOUT_ERROR : 45|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ WARN_KL15_UNDER_V : 59|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ WARN_BRAKE_PLAUSIBILITY : 58|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ WARN_AI_COMMS_LOST : 61|1@1+ (1,0) [0|1] "" Vector__XXX