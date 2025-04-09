This demo uses an ML Training application for image classification that consumes a protected ML model,
trains it on the data from another party and writes the trained model to the output.
The sample flows build the initial untrained ML model and safely protects in the fabrickam model input storage.
It also downloads and stores the sample training data to Contosso's storage account to be used in the cleanroom
as the confidential training data. It then builds the application container image with the required dependencies
and push it into an Azure Container Registry. This container would be used to load the untrained model and
training data to train the model and write the output model to fabrikam's storage.
When the cleanroom starts, this application is auto triggered to start the training of the model.

| Persona   | Input                 | Output                    |
| :---      | :---                  | :---                      |
| litware   | Application           | Telemetry                 |
| fabrikam  | Model                 | Trained model             |
| contosso  | Data (Data source)    | N/A                       |

This sample has been adopted from https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model

The clean room infrastructure abstracts away the encryption and confidential computation details, allowing the application to access the model and data sets in clear text as if executing in a regular container environment.