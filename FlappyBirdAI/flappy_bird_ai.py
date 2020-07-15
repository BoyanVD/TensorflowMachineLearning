import os
from NEAT_Algorithm import run

def main():
    local_directory = os.path.dirname(__file__)
    configuration_path = os.path.join(local_directory, "config-feedforward.txt")
    run(configuration_path)

if __name__ == "__main__":
    main()
