import json


g = 9.81


#%% Peugeot 205 GTi RFS
with open('./cars/Peugeot_205RFS.json', 'w') as fp:
    json.dump(dict(
            name = 'Peugeot 205 Gti RFS', 
            acc_grip_max = 1.15 * g,
            acc_limit = 0.4455 * g, 
            dec_limit = 1.00 * g, 
            mass = 1045,
            P_engine = 122,   #bhp  (108hp@wheels)
            c_drag = 0.5 * 0.28 * 1.21 * 1.58,        # ref: http://www.carinf.com/en/b41047154.html
            c_roll = 0.016,  #approximation (low)
            trail_braking = 70, #percentage trialbraking, based on driver expertise
        ), fp, sort_keys=True, indent=4)


#%% BMW Z3M Viperwizard
with open('./cars/BMW_Z3M.json', 'w') as fp:
    json.dump(dict(
            name = 'BMW Z3M Viperwizard',
            acc_grip_max = 1.35 * g,
            acc_limit = 0.33 * 1.35 * g, 
            dec_limit = 0.85 * 1.35 * g, 
            mass = 1450,
            P_engine = 208,             #bhp  (228hp@wheels)
            c_drag = 0.5 * 0.37 * 1.22 * 1.83,        # ref:http://www.carinf.com/en/ff3031158.html
            c_roll = 0.015, #approximation
            trail_braking = 40, #percentage trialbraking, based on driver expertise
        ), fp, sort_keys=True, indent=4)
    
