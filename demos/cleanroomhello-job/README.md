This demo uses a simple application that reads a protected file from an input path and writes a protected compressed file to an output path.

The clean room infrastructure abstracts away the encryption and confidential computation details, allowing the application to process data in clear text as if executing in a regular container environment.

The demo application is injected as inline code passed in the form of a command to the official `golang` container.

The sample flows download the compressed file after the clean room has completed execution and display the decompressed content - this is expected to be identical to the original input.

| Persona   | Input         | Output    |
| :---      | :---          | :---      |
| litware   | Application   | Telemetry |
| fabrikam  | Data          | Data      |
| contosso  | NA            | NA        |
| client    | NA            | NA        |

<!-- TODO: Enhance cleanroomhello-job demo application.

    litware - application in, logs and telemetry out
    fabrikam - data in, data out
    contosso - data in, data out
    consumer - NA
-->