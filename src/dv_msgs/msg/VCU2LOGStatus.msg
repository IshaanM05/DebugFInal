uint8 state_assi              # Autonomous System Status Indicator (0 to 7)
uint8 state_ebs               # Emergency Brake System State (0 to 3)
uint8 ami_state               # Autonomous Mode Indicator State (0 to 7)
bool state_steering           # Steering State (0 or 1)
uint8 state_service_brake     # Service Brake State (0 to 3)
uint8 lap_counter             # Lap Counter (0 to 15)
uint8 cones_count_actual      # Actual Cones Count (0 to 255)
uint32 cones_count_all        # Total Cones Count (0 to 131071)

# BO_ 1282 VCU2LOG_Status: 5 VCU
#  SG_ State_ASSI : 0|3@1+ (1,0) [0|7] "" Vector__XXX
#  SG_ State_EBS : 3|2@1+ (1,0) [0|3] "" Vector__XXX
#  SG_ AMI_STATE : 5|3@1+ (1,0) [0|7] "" Vector__XXX
#  SG_ State_steering : 8|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ State_service_brake : 9|2@1+ (1,0) [0|3] "" Vector__XXX
#  SG_ Lap_counter : 11|4@1+ (1,0) [0|15] "" Vector__XXX
#  SG_ Cones_count_actual : 15|8@1+ (1,0) [0|255] "" Vector__XXX
#  SG_ Cones_count_all : 23|17@1+ (1,0) [0|131071] "" Vector__XXX