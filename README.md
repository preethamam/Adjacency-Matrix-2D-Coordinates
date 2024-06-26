# Description
Creates a weighted adjacency matrix for any 2D (in fact the same program works for `N-dimension`, just create an `N-dimensional` input vector/matrix) coordinates of rectangular or non-rectangular grids based on the 8-noded nearest neigbors. Each coordinate represents a node in a graph at that position which is connected to other nodes if they are adjacent in the x or y direction, and diagonals are also included.

## Graphs table
| Type | Images |
| --- | --- |
| 8-noded undirected | <img width="541" alt="Screenshot 2024-03-29 at 2 43 12 AM" src="https://github.com/preethamam/Adjacency-Matrix-2D-Coordinates/assets/28588878/2b58a372-47f3-4312-871a-33354eb196e4"> |
| 8-noded directed | <img width="542" alt="Screenshot 2024-03-29 at 2 41 09 AM" src="https://github.com/preethamam/Adjacency-Matrix-2D-Coordinates/assets/28588878/b24eeef9-a3e8-4236-b28e-56e19c333b44"> |


## Installation
```shell
pip install -r requirements.txt
```

## Requirements
```python
Python >= 3.10.13
joblib==1.3.2
matplotlib==3.8.3
networkx==3.2.1
numpy==1.26.4
scipy==1.12.0
tqdm==4.66.2
```

## Usage
```shell
cd /path/to/the/Adjaceny Matrix folder
python adjacency_matrix.py
```
Please use the `adjacency_matrix.py` to run the program.

## Feedback
Please rate and provide feedback for the further improvements.
