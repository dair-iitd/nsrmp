# Data Generation

This code is used to generate te data used for training and testing the model.

## Setup and Execution

Install the packages from requirement.txt and run construct.sh file from the DataGeneration_PyBullet directory.

```bash
cd ./DataGeneration_PyBullet && bash construct.sh
```

## Description of the pipeline

### templates.json

This code loads the program and instruction templates from json files and uses them to create program, objects and instructions.
The  program template has the following pattern.
```
{
        "depth": 1,
        "text": {
            "simple": [
                "Put the <C1> <T1> <A1> <C2> <T2>" 
            ],
            "complex":[
                "put the <T1> which is <C1> in color <A1> <C2> <T2>",
                "put the <T1> that is <C1> in color <A1>  <T2> which is <C2> in color"

                
            ],
            "compound":[
                "put the <T1> whose color  is <C1>  <A1> block which is a <T2> and <C2> in color"
            ]
        },
        "program": [
            {
                "action": "<A1>",
                "inputs": "<O1> <O2>"
            }
        ],
        "constraints": {
            "actions": [],
            "inputs": [
                1,
                2
            ]
        },
        "num_unique_actions": 1,
        "num_unique_objects": 2
}
``` 
- **depth**: No of actions or subtasks in the program  
- **text**: Template for the question  
- **program**: Template for the program. Each subtask in the program takes an action and two objects as its input.  
- **constraints**: This tells us which objects and actions should be unique. If the list is empty, no such uniqueness constraints is imposed.  
- **Tokens**: T - type, C - color, A - actions, O - objects  R- Relational Attributes, M- Material, S-size  

### metadata.json

This file has synonyms and the concept words. During instructruction generation, the words are randomly replaced with thier synonyms to create variations in the language. 

### configs.py 

The configs.py file helps us to control the data generation by providing AdditionalConfigs. We can control the rotation, complexity of the natural language instruction, number and tyoes objects in the scene, etc.

### Objects and Types
Currently, the code supports the generation and manipulation of the following Concepts/Attributes.

- **Objects**: Small Cube, Lego block, Dice  
- **Colors**: Blue, Green, Yellow, Magenta, Red, Cyan, White  
- **Actions**: MoveTOP, MoveLEFT, MoveRight  


