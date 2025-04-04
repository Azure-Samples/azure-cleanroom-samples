
This demo uses an ML Training application for image classification that consumes a protected ML model, trains it on the data from another party and writes the trained model to the output.

This sample has been adopted from https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model

The clean room infrastructure abstracts away the encryption and confidential computation details, allowing the application to access the model and data sets in clear text as if executing in a regular container environment.