from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8020), SimpleHTTPRequestHandler)
    print("Model visualizer UI running at http://127.0.0.1:8020")
    server.serve_forever()


if __name__ == "__main__":
    main()
