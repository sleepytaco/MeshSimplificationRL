
Agent V1 is meant to be the "proof-of-concept" version of the project. I make the assumption of working with a fixed size mesh (500 faces -> 750 edges). I have modelled the policy to select which one of the 
- **States**: A N x 3 matrix that contains vertices in the first half and faces in the second. Here, N = number of vertices + number of faces.
- **Actions**: I will let the agent pick which edge it wants to collapse from all possible edge IDs — the actions space would be in the form of a M x 1 dimensional vector where M is the number of edges in the original mesh.
- **Rewards**: 
  - I’d like to start off by rewarding the agent’s choice of edge to collapse with the QEM cost for the edge. 
  - I will not give the agent any reward if it chooses an already collapsed edge.
  - I will give the agent a negative reward if it collapses an edge that will result in the mesh being non-manifold, or that will end up inverting the nearby faces.
- **Transitions**: There is no need for a transition probability here due to the deterministic nature of edge removal from a mesh. We will know exactly how the resulting mesh will look like once I collapse an edge with a particular edge ID.

I have used TRPO to train the agent. I found that with such a set up the agent has learned to generalize to a mesh that it was not trained on. For example, lets say the agent was trained on a bunny mesh to simplify from a 500-face mesh to a 100-face mesh --- it learned to simplify it while preserving the bunny's ears, face, and body. Then, when I test the agent with a cow mesh, it is able to preserve the cows legs, face, and body. 

While it is good that it is able to generalize, this version of the agent has the following issues:
- The agent picks edges that were already collapsed most of the time. It does not learn when certain edges have collapsed.
- The agent looses the sense of the volume of the object when it is asked to simplify a mesh down to below 150 faces.
- The agent is fixed to a 500-faced mesh as a starting point.
- The agent relies on a standard MLP for the policy. I will upgrade to MeshCNN (which is like CNN but for meshes) as the feature extractor.