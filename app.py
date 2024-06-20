"""
This module uses fuzzy logic to evaluate individual compatibility based on various features,
leveraging the skfuzzy and numpy libraries. It defines linguistic variables, membership functions,
and rules to compute compatibility scores for:

    Morphological/Physiological features (skin color, hair color, height)
    Psychological features (temperament)
    Medical clues (blood group)
    Salary/Earnings features (educational level)

Fuzzy rules for each feature category are combined into control systems to compute an overall compatibility score.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define linguistic variable for compatibility
compatibility = ctrl.Consequent(np.arange(0, 101, 1), 'compatibility')
compatibility['low'] = fuzz.trimf(compatibility.universe, [0, 25, 50])
compatibility['medium'] = fuzz.trimf(compatibility.universe, [25, 50, 75])
compatibility['high'] = fuzz.trimf(compatibility.universe, [50, 75, 100])

# Define linguistic variables and membership functions for Morphological/Physiological features
skin_colour = ctrl.Antecedent(np.arange(0, 101, 1), 'skin_colour')
skin_colour['light'] = fuzz.trimf(skin_colour.universe, [0, 25, 50])
skin_colour['medium'] = fuzz.trimf(skin_colour.universe, [25, 50, 75])
skin_colour['dark'] = fuzz.trimf(skin_colour.universe, [50, 75, 100])

hair_colour = ctrl.Antecedent(np.arange(0, 101, 1), 'hair_colour')
hair_colour['blonde'] = fuzz.trimf(hair_colour.universe, [0, 25, 50])
hair_colour['brown'] = fuzz.trimf(hair_colour.universe, [25, 50, 75])
hair_colour['black'] = fuzz.trimf(hair_colour.universe, [50, 75, 100])

height = ctrl.Antecedent(np.arange(120, 220, 1), 'height')
height['short'] = fuzz.trimf(height.universe, [120, 150, 170])
height['average'] = fuzz.trimf(height.universe, [150, 170, 190])
height['tall'] = fuzz.trimf(height.universe, [170, 190, 220])

# Define linguistic variables and membership functions for Psychological features
temperament = ctrl.Antecedent(np.arange(0, 101, 1), 'temperament')
temperament['calm'] = fuzz.trimf(temperament.universe, [0, 25, 50])
temperament['balanced'] = fuzz.trimf(temperament.universe, [25, 50, 75])
temperament['angry'] = fuzz.trimf(temperament.universe, [50, 75, 100])

# Define linguistic variables and membership functions for Medical clues features
blood_group = ctrl.Antecedent(np.arange(0, 101, 1), 'blood_group')
blood_group['A'] = fuzz.trimf(blood_group.universe, [0, 25, 50])
blood_group['B'] = fuzz.trimf(blood_group.universe, [25, 50, 75])
blood_group['O'] = fuzz.trimf(blood_group.universe, [50, 75, 100])

# Define linguistic variables and membership functions for salary/earnings features
educational_level = ctrl.Antecedent(np.arange(0, 101, 1), 'educational_level')
educational_level['low'] = fuzz.trimf(educational_level.universe, [0, 25, 50])
educational_level['medium'] = fuzz.trimf(educational_level.universe, [25, 50, 75])
educational_level['high'] = fuzz.trimf(educational_level.universe, [50, 75, 100])

# Define rules for compatibility based on Morphological/Physiological features
rule_skin_light = ctrl.Rule(skin_colour['light'] & hair_colour['blonde'] & height['tall'], compatibility['high'])
rule_skin_medium = ctrl.Rule(skin_colour['medium'] & hair_colour['brown'] & height['average'], compatibility['medium'])
rule_skin_dark = ctrl.Rule(skin_colour['dark'] & hair_colour['black'] & height['short'], compatibility['low'])

# Define rules for compatibility based on Psychological features
rule_calm = ctrl.Rule(temperament['calm'], compatibility['high'])
rule_balanced = ctrl.Rule(temperament['balanced'], compatibility['medium'])
rule_angry = ctrl.Rule(temperament['angry'], compatibility['low'])

# Define rules for compatibility based on Medical clues features
rule_blood_A = ctrl.Rule(blood_group['A'], compatibility['high'])
rule_blood_B = ctrl.Rule(blood_group['B'], compatibility['medium'])
rule_blood_O = ctrl.Rule(blood_group['O'], compatibility['low'])

# Define rules for compatibility based on salary/earnings features
rule_educ_low = ctrl.Rule(educational_level['low'], compatibility['low'])
rule_educ_medium = ctrl.Rule(educational_level['medium'], compatibility['medium'])
rule_educ_high = ctrl.Rule(educational_level['high'], compatibility['high'])

# Create control systems
compatibility_morphological_ctrl = ctrl.ControlSystem([rule_skin_light, rule_skin_medium, rule_skin_dark])
compatibility_psychological_ctrl = ctrl.ControlSystem([rule_calm, rule_balanced, rule_angry])
compatibility_medical_ctrl = ctrl.ControlSystem([rule_blood_A, rule_blood_B, rule_blood_O])
compatibility_salary_ctrl = ctrl.ControlSystem([rule_educ_low, rule_educ_medium, rule_educ_high])

def compute_compatibility_score(skin_color, hair_color, height, temperament, blood_group, educational_level):
    """
    Compute the compatibility score for an individual based on the input features.

    Parameters:
    skin_color (int): Skin color value (0-100)
    hair_color (int): Hair color value (0-100)
    height (int): Height value (120-220)
    temperament (int): Temperament value (0-100)
    blood_group (int): Blood group value (0-100)
    educational_level (int): Educational level value (0-100)

    Returns:
    float: Overall compatibility score
    """
    morphological_sim = ctrl.ControlSystemSimulation(compatibility_morphological_ctrl)
    psychological_sim = ctrl.ControlSystemSimulation(compatibility_psychological_ctrl)
    medical_sim = ctrl.ControlSystemSimulation(compatibility_medical_ctrl)
    salary_sim = ctrl.ControlSystemSimulation(compatibility_salary_ctrl)

    morphological_sim.input['skin_colour'] = skin_color
    morphological_sim.input['hair_colour'] = hair_color
    morphological_sim.input['height'] = height
    psychological_sim.input['temperament'] = temperament
    medical_sim.input['blood_group'] = blood_group
    salary_sim.input['educational_level'] = educational_level

    morphological_sim.compute()
    psychological_sim.compute()
    medical_sim.compute()
    salary_sim.compute()

    overall_compatibility_score = (morphological_sim.output['compatibility'] +
                                   psychological_sim.output['compatibility'] +
                                   medical_sim.output['compatibility'] +
                                   salary_sim.output['compatibility']) / 4
    
    return overall_compatibility_score

# Example inputs for a pair of individuals
skin_color_male = 70
hair_color_male = 60
height_male = 180
temperament_male = 20
blood_group_male = 30
educational_level_male = 80

skin_color_female = 80
hair_color_female = 70
height_female = 165
temperament_female = 70
blood_group_female = 40
educational_level_female = 70

compatibility_score_male = compute_compatibility_score(skin_color_male, hair_color_male, height_male,
                                                       temperament_male, blood_group_male, educational_level_male)

compatibility_score_female = compute_compatibility_score(skin_color_female, hair_color_female, height_female,
                                                         temperament_female, blood_group_female, educational_level_female)

print("Compatibility score for male:", compatibility_score_male)
print("Compatibility score for female:", compatibility_score_female)

if compatibility_score_male >= 50 and compatibility_score_female >= 50:
    print("Overall compatibility as a couple: High")
else:
    print("Overall compatibility as a couple: Low")
