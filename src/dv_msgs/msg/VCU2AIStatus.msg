bool as_switch_status        # Autonomous Switch Status
bool ts_switch_status        # Traction Switch Status
uint8 as_state               # Autonomous State (0-7)
bool steering_status         # Steering Status 
bool shutdown_request        # Shutdown Request
bool fault_status            # Fault Status
bool warning_status          # Warning Status
bool handshake               # Handshake Status
uint8 ami_state              # Autonomous Mode Indicator State (0-15)
bool go_signal               # Go Signal
bool ai_estop_request        # AI Emergency Stop Request
bool hvil_open_fault         # High Voltage Interlock Loop Open Fault
bool hvil_short_fault        # High Voltage Interlock Loop Short Fault
bool ebs_fault               # Emergency Brake System Fault
bool offboard_charger_fault  # Offboard Charger Fault
bool ai_comms_lost           # AI Communications Lost
bool warn_batt_temp_high     # Battery Temperature High Warning
bool warn_batt_soc_low       # Battery State of Charge Low Warning
bool charge_procedure_fault  # Charge Procedure Fault
bool autonomous_braking_fault # Autonomous Braking Fault
bool mission_status_fault    # Mission Status Fault
uint8 shutdown_cause         # Shutdown Cause (0-255)
bool bms_fault               # Battery Management System Fault
bool brake_plausibility_fault # Brake Plausibility Fault

# BO_ 1312 VCU2AI_Status: 8 VCU
#  SG_ AS_SWITCH_STATUS : 9|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ TS_SWITCH_STATUS : 10|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ AS_STATE : 16|4@1+ (1,0) [0|7] "" Vector__XXX
#  SG_ STEERING_STATUS : 12|2@1+ (1,0) [0|3] "" Vector__XXX
#  SG_ SHUTDOWN_REQUEST : 8|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ FAULT_STATUS : 24|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ WARNING_STATUS : 25|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ HANDSHAKE : 0|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ AMI_STATE : 20|4@1+ (1,0) [0|15] "" Vector__XXX
#  SG_ GO_SIGNAL : 11|1@1+ (1,0) [0|0] "" Vector__XXX
#  SG_ AI_ESTOP_REQUEST : 40|1@1+ (1,0) [0|1] "__F__" Vector__XXX
#  SG_ HVIL_OPEN_FAULT : 41|1@1+ (1,0) [0|1] "__F__" Vector__XXX
#  SG_ HVIL_SHORT_FAULT : 42|1@1+ (1,0) [0|1] "__F__" Vector__XXX
#  SG_ EBS_FAULT : 43|1@1+ (1,0) [0|1] "__F__" Vector__XXX
#  SG_ OFFBOARD_CHARGER_FAULT : 44|1@1+ (1,0) [0|1] "__F__" Vector__XXX
#  SG_ AI_COMMS_LOST : 45|1@1+ (1,0) [0|1] "__F__" Vector__XXX
#  SG_ WARN_BATT_TEMP_HIGH : 32|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ WARN_BATT_SOC_LOW : 33|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ CHARGE_PROCEDURE_FAULT : 48|1@1+ (1,0) [0|1] "__F__" Vector__XXX
#  SG_ AUTONOMOUS_BRAKING_FAULT : 46|1@1+ (1,0) [0|1] "__F__" Vector__XXX
#  SG_ MISSION_STATUS_FAULT : 47|1@1+ (1,0) [0|1] "__F__" Vector__XXX
#  SG_ SHUTDOWN_CAUSE : 56|8@1+ (1,0) [0|255] "" Vector__XXX
#  SG_ BMS_FAULT : 49|1@1+ (1,0) [0|1] "__F__" Vector__XXX
#  SG_ BRAKE_PLAUSIBILITY_FAULT : 50|1@1+ (1,0) [0|1] "__F__" Vector__XXX
