This demo uses an inference application that consumes a protected ML `onnx` model and provides an API endpoint for infererring _sentiment_ of a movie review.

The clean room infrastructure abstracts away the encryption and confidential computation details, allowing the application to access the model and data sets in clear text as if executing in a regular container environment.

The sample flows build the initial ML model and inject it as data, build the application container image with the required dependencies and push it into an Azure Container Registry, and invoke the API endpoint after the clean room has been deployed, displaying _inferred_ sentiment for positive and negative reviews.

| Persona   | Input                 | Output                    |
| :---      | :---                  | :---                      |
| litware   | Application           | Telemetry                 |
| fabrikam  | Model                 | NA                        |
| contosso  | Data (Data source)    | Inference (API Response)  |
| client    | Data (API Payload)    | Inference (API Response)  |

> [!NOTE]
> The sample flow steps for constructing the initial model and preparing the application may take a long time to complete as multiple dependencies need to be installed.

> [!NOTE]
> The sample flows inject a protected data set for `contosso` to calculate accuracy of inference. However, the application API is not functioning as expected and is currently being investigated.