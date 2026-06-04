import re

with open("lens_test_server.py", "r") as f:
    text = f.read()

# Make sure all variables passed to round() in dictionaries are explicitly cast to float
# to prevent "Object of type float32 is not JSON serializable"
text = re.sub(r'round\(([^,]+?),\s*4\)', r'round(float(\1), 4)', text)
text = re.sub(r'round\(([^,]+?),\s*6\)', r'round(float(\1), 6)', text)
text = re.sub(r'round\(([^,]+?),\s*2\)', r'round(float(\1), 2)', text)

# Just in case, let's also cast cfa_ratio manually where it's declared to be safe
text = text.replace('cfa_ratio = min(var_h_even, var_h_odd) / max(var_h_even, var_h_odd)', 
                    'cfa_ratio = float(min(var_h_even, var_h_odd) / max(var_h_even, var_h_odd))')

with open("lens_test_server.py", "w") as f:
    f.write(text)
print("Type casting applied.")
