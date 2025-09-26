uint8 speed_actual           # Actual speed in km/h (0 to 255)
uint8 speed_target           # Target speed in km/h (0 to 255)
float32 steer_actual         # Actual steering angle in degrees (-64 to 63.5)
float32 steer_target         # Target steering angle in degrees (-64 to 63.5)
uint8 brake_actual_pct       # Actual brake percentage (0 to 100)
uint8 brake_target_pct       # Target brake percentage (0 to 100)
uint8 drive_trq_actual_pct   # Actual drive torque percentage (0 to 100)
uint8 drive_trq_target_pct   # Target drive torque percentage (0 to 100)

# BO_ 1280 VCU2LOG_Dynamics1: 8 VCU
#  SG_ Speed_actual : 0|8@1+ (1,0) [0|255] "km/h" Vector__XXX
#  SG_ Speed_target : 8|8@1+ (1,0) [0|255] "km/h" Vector__XXX
#  SG_ Steer_actual : 16|8@1- (0.5,0) [-64|63.5] "deg" Vector__XXX
#  SG_ Steer_target : 24|8@1- (0.5,0) [-64|63.5] "deg" Vector__XXX
#  SG_ Brake_actual_pct : 32|8@1+ (1,0) [0|100] "%" Vector__XXX
#  SG_ Brake_target_pct : 40|8@1+ (1,0) [0|100] "%" Vector__XXX
#  SG_ Drive_trq_actual_pct : 48|8@1+ (1,0) [0|100] "%" Vector__XXX
#  SG_ Drive_trq_target_pct : 56|8@1+ (1,0) [0|100] "%" Vector__XXX