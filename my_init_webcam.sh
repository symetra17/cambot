v4l2-ctl -d /dev/video0 --set-ctrl=brightness=128
v4l2-ctl -d /dev/video0 --set-ctrl=contrast=128
v4l2-ctl -d /dev/video0 --set-ctrl=saturation=128
v4l2-ctl -d /dev/video0 --set-ctrl=sharpness=128
v4l2-ctl -d /dev/video0 --set-ctrl=power_line_frequency=0
v4l2-ctl -d /dev/video0 --set-ctrl=backlight_compensation=0

# exposure_auto=1 for manual
# exposure_auto=3 for auto
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=3
#v4l2-ctl -d /dev/video0 --set-ctrl=exposure_absolute=700
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto_priority=0

v4l2-ctl -d /dev/video0 --set-ctrl=gain=50
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_temperature_auto=0
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_temperature=4200
v4l2-ctl -d /dev/video0 --set-ctrl=focus_auto=0
v4l2-ctl -d /dev/video0 --set-ctrl=focus_absolute=0
v4l2-ctl -d /dev/video0 --set-ctrl=zoom_absolute=120

# list all supported format:
#v4l2-ctl -d /dev/video0 --list-formats-ext

# list all controls:
#v4l2-ctl -d /dev/video0 --list-ctrls-menu

# list current video format:
# v4l2-ctl -V

# settings for Logitech BRIO
#
#                     brightness 0x00980900 (int)    : min=0 max=255 step=1 default=128 value=128
#                       contrast 0x00980901 (int)    : min=0 max=255 step=1 default=128 value=128
#                     saturation 0x00980902 (int)    : min=0 max=255 step=1 default=128 value=128
# white_balance_temperature_auto 0x0098090c (bool)   : default=1 value=1
#                           gain 0x00980913 (int)    : min=0 max=255 step=1 default=0 value=0
#           power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=2 value=2
#				0: Disabled
#				1: 50 Hz
#				2: 60 Hz
#      white_balance_temperature 0x0098091a (int)    : min=2000 max=7500 step=10 default=4000 value=4760 flags=inactive
#                      sharpness 0x0098091b (int)    : min=0 max=255 step=1 default=128 value=128
#         backlight_compensation 0x0098091c (int)    : min=0 max=1 step=1 default=1 value=1
#                  exposure_auto 0x009a0901 (menu)   : min=0 max=3 default=3 value=3
#				1: Manual Mode
#				3: Aperture Priority Mode
#              exposure_absolute 0x009a0902 (int)    : min=3 max=2047 step=1 default=250 value=312 flags=inactive
#         exposure_auto_priority 0x009a0903 (bool)   : default=0 value=1
#                   pan_absolute 0x009a0908 (int)    : min=-36000 max=36000 step=3600 default=0 value=0
#                  tilt_absolute 0x009a0909 (int)    : min=-36000 max=36000 step=3600 default=0 value=0
#                 focus_absolute 0x009a090a (int)    : min=0 max=255 step=5 default=0 value=10 flags=inactive
#                     focus_auto 0x009a090c (bool)   : default=1 value=1
#                  zoom_absolute 0x009a090d (int)    : min=100 max=500 step=1 default=100 value=100

