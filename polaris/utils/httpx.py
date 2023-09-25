from httpx import Response


def _log_response(response: Response) -> str:
    """
    Fully logs a request/response pair for HTTPX.
    Used for debugging purposes.
    """
    req_prefix = "< "
    res_prefix = "> "
    request = response.request
    output = [f"{req_prefix}{request.method} {request.url}"]

    for name, value in request.headers.items():
        output.append(f"{req_prefix}{name}: {value}")

    output.append(req_prefix)

    if isinstance(request.content, (str, bytes)):
        output.append(f"{req_prefix}{request.content}")
    else:
        output.append("<< Request body is not a string-like type >>")

    output.append("")

    output.append(f"{res_prefix} {response.status_code} {response.reason_phrase}")

    for name, value in response.headers.items():
        output.append(f"{res_prefix}{name}: {value}")

    output.append(res_prefix)

    output.append(f"{res_prefix}{response.text}")

    return "\n".join(output)
