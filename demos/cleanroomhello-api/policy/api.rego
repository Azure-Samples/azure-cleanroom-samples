package ccr.policy

import future.keywords

default on_request_headers = {
    "allowed": false,
    "http_status": 403,
    "body": {
        "code": "RequestNotAllowed",
        "message": "Failed ccr policy check: Requested API is not allowed"
    }
}

default on_request_body = {
    "allowed": false,
    "http_status": 403,
    "body": {
        "code": "RequestBodyNotAllowed",
        "message": "Failed ccr policy check: Requested API body is not allowed"
    }
}

on_request_headers = result {
    get_method == "GET"
    get_path == "/"
    result := {
        "allowed": true
    }
}

default on_response_body = true
default on_response_headers = true

get_path := path if {
    some header in input.requestHeaders.headers.headers
    header.key == ":path"
    path := header.value
} else := ""

get_method := method if {
    some header in input.requestHeaders.headers.headers
    header.key == ":method"
    method := header.value
} else := ""

