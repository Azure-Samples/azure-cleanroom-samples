This demo uses the default `nginx-hello` application to bring up an API endpoint for servicing network calls.

Network protection policy is associated with the endpoint that determines the allowed API calls. The sample flows package this policy into an OCI artefact and push it into an Azure Container Registry

The sample flows invoke the API endpoint after the clean room has been deployed, displaying response from both allowed and disallowed calls.

| Persona   | Input                         | Output    |
| :---      | :---                          | :---      |
| litware   | Application, Network Policy   | Telemetry |
| fabrikam  | NA                            | NA        |
| contosso  | NA                            | NA        |
| client    | NA                            | NA        |

<!-- TODO: Enhance cleanroomhello-api demo application.

    litware - application in, logs and telemetry out
    fabrikam - data in, nothing out
    contosso - data in, nothing out
    consumer - API request in, API response out
-->