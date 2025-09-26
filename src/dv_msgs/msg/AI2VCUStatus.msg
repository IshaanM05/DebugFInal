uint8 direction_request    # Direction request (0 to 3)
bool estop_request         # Emergency stop request
uint8 mission_status       # Mission status (0 to 3)
bool handshake             # Handshake status
uint8 lap_counter          # Lap counter (0 to 15)
uint8 cones_count_actual   # Actual count of cones (0 to 255)
uint16 cones_count_all     # Total count of cones (0 to 65535)
uint8 veh_speed_actual     # Actual vehicle speed in km/h (0 to 255)
uint8 veh_speed_demand     # Demanded vehicle speed in km/h (0 to 255)

# BO_ 1296 AI2VCU_Status: 8 AI
#  SG_ DIRECTION_REQUEST : 14|2@1+ (1,0) [0|3] "" Vector__XXX
#  SG_ ESTOP_REQUEST : 8|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ MISSION_STATUS : 12|2@1+ (1,0) [0|3] "" Vector__XXX
#  SG_ HANDSHAKE : 0|1@1+ (1,0) [0|1] "" Vector__XXX
#  SG_ LAP_COUNTER : 16|4@1+ (1,0) [0|15] "" Vector__XXX
#  SG_ CONES_COUNT_ACTUAL : 24|8@1+ (1,0) [0|255] "" Vector__XXX
#  SG_ CONES_COUNT_ALL : 32|16@1+ (1,0) [0|65535] "" Vector__XXX
#  SG_ VEH_SPEED_ACTUAL : 48|8@1+ (1,0) [0|255] "km/h" Vector__XXX
#  SG_ VEH_SPEED_DEMAND : 56|8@1+ (1,0) [0|255] "km/h" Vector__XXX