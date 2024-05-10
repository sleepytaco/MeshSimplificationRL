# Mesh Simplification using RL

**TL;DR** I am attempting to model mesh simplification as a reinforcement learning problem to train an agent to look at a dense mesh and learn to remove edges in a way that results in a simplified mesh that still holds a geometric resemblance with the original dense mesh. I want to investigate how well an RL-based mesh simplification agent can learn a policy to do the task while making sure to not collapse edges that would lead to the resulting mesh being non-manifold. In the process, I also want to explore how well the agent can generalize to arbitrary fixed-size meshes.

I implemented the quadric error mesh simplification algorithm in C++ and built an RL environment interface around it. I built a simple C++ API server that will allow my Python program to interact with the mesh simplification environment through the endpoints. I made a custom Gym environment in Python that calls the API endpoints to reset the environment, take actions in the environment, return the current state, etc. I experimented with a naive MDP formulation to see how well the agent does. This project is still in progress!

Link to project presentation slides: https://docs.google.com/presentation/d/1hqYyW1kb2VyasSK4ohizyUo9dW6VpY9fVQkz_RlPaWo/edit?usp=sharing


## Background

Meshes are the go-to way to represent and discretize an arbitrary 3D surface in computer graphics. A mesh is defined by a set of vertices and a set of faces. Each face is described by a list of three vertex IDs that are linked in a counter-clockwise fashion. One common mesh processing operation is mesh simplification, which involves starting from a densely connected mesh and ending up with a “simplified” version of the original mesh. The goal of mesh simplification is to remove edges from the original mesh in a sequential fashion such that the core geometric details from the original mesh are preserved. For example, if I start with a dense skull mesh, I hope to end up with a simplified mesh that still (remotely) resembles the characteristics of the original skull mesh. The number of triangles/faces that you would like to remove is often passed as a parameter to the mesh simplification algorithm.

A very traditional and popular mesh simplification algorithm in computer graphics literature utilizes what is called the quadric error metric (QEM). The idea is to assign each edge in the mesh a heuristic cost based on how badly the surface would change if that edge were to be removed. This is called the quadric error cost. Once all edges have been assigned this QEM cost, the algorithm simply picks edges with a low QEM cost and removes them from the mesh one by one. Note that every time an edge is removed (or, collapsed) from the mesh the QEM costs of the nearby edges will have to be updated to reflect the new mesh geometry.

One property of meshes I’d like to introduce is manifoldness. The manifold property for meshes specifies that the mesh is a closed surface and ensures that it has nice edge connectivity. The reason this is commonly used in computer graphics is that manifold meshes make mesh processing easier by letting us not worry about weird edge cases that might arise (no pun intended). All meshes I work with will have this manifoldness property.

## Project Goals

The mesh simplification task has an inherent sequential nature to it. The order in which you remove edges from the mesh matters. If you were to blindly remove edges one by one, you might end up with a non-manifold mesh, or might inadvertently flip the orientation of the triangles in the mesh — both of which give grotesque results. One issue with the traditional QEM algorithm is that, since it assigns a cost to each edge and greedily removes the low-cost edges, it has no idea if collapsing a particular edge will lead to non-manifoldness or unintentionally inverting nearby triangles. The way we deal with this problem in code is to manually check if the edge collapse results in non-manifoldness or inverting of nearby triangles — if it does, we ignore that edge collapse operation and move on to the next edge with the lowest cost.

I propose to model mesh simplification as a reinforcement learning problem to train an agent to look at a dense mesh and learn to remove edges in a way that results in a simplified mesh that still holds a geometric resemblance with the original dense mesh.

Here is a starting point on how I’m planning to model the mesh simplification problem as an MDP:
- **States**: A N x 3 matrix that contains vertices in the first half and faces in the second. Here, N = number of vertices + number of faces.
- **Actions**: I will let the agent pick which edge it wants to collapse from all possible edge IDs — the actions space would be in the form of a M x 1 dimensional vector where M is the number of edges in the original mesh.
- **Rewards**: 
  - I’d like to start off by rewarding the agent’s choice of edge to collapse with the QEM cost for the edge. 
  - I will not give the agent any reward if it chooses an already collapsed edge.
  - I will give the agent a negative reward if it collapses an edge that will result in the mesh being non-manifold, or that will end up inverting the nearby faces.
- **Transitions**: There is no need for a transition probability here due to the deterministic nature of edge removal from a mesh. We will know exactly how the resulting mesh will look like once I collapse an edge with a particular edge ID.

