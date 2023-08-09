import subprocess
import sys

# Get the value of K from the command line argument
if len(sys.argv) < 2:
    print("Usage: python run_demo.py <K>")
    sys.exit(1)

k_value = int(sys.argv[1])

# Run Part 1 script with K parameter
subprocess.run(["python", "recommendation_demo.py", "--k", str(k_value)])

# Run Part 2 script
subprocess.run(["python", "causal_distillation_demo.py"])
