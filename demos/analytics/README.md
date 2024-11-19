This demo uses an analytics application that consumes protected data using `spark` engine and provides an API endpoint for executing audited queries.

The clean room infrastructure abstracts away the encryption and confidential computation details, allowing the application to present data sets to the spark engine in clear text as if executing in a regular container environment.

The clean room infrastructure abstracts away the governance infrastructure and CCF details, allowing the application to get the audited queries from a localhost endpoint.

The sample flows build the application container image with the required dependencies and push it into an Azure Container Registry, register allowed queries with the governance service, and invoke the API endpoint after the clean room has been deployed to execute them.

| Persona   | Input         | Output    |
| :---      | :---          | :---      |
| litware   | Application   | Telemetry |
| fabrikam  | Data, Query   | NA        |
| contosso  | Data, Query   | NA        |
| client    | NA            | NA        |

<!-- TODO: Enhance analytics demo application.
    fabrikam - Enhance queries to produce more insightful output.
    fabrikam - Enhance queries to produce more insightful output.
    client - API request query-id, query-params in, API response query-result out
-->
