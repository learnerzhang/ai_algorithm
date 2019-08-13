#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-18 17:50
# @Author  : zhangzhen
# @Site    : 
# @File    : ssr2json.py
# @Software: PyCharm
import argparse
import collections
from typing import Text, Any, List
from pprint import pprint
import base64
import json
import os
import io

ssr_template = {
    "index": 0,
    "random": False,
    "global": False,
    "enabled": True,
    "shareOverLan": False,
    "isDefault": False,
    "localPort": 1080,
    "pacUrl": None,
    "useOnlinePac": False,
    "reconnectTimes": 3,
    "randomAlgorithm": 0,
    "TTL": 0,
    "proxyEnable": False,
    "proxyType": 0,
    "proxyHost": None,
    "proxyPort": 0,
    "proxyAuthUser": None,
    "proxyAuthPass": None,
    "authUser": None,
    "authPass": None,
    "autoban": False,
    "configs": []
}


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="ssr file path")
    parser.add_argument("--output", type=str, default="shadowsocksR.json", help="output config file path")

    return parser.parse_known_args()


FLAGS, unparsed = args()


def write_json_to_file(filename: Text, obj: Any, **kwargs: Any) -> None:
    write_to_file(filename, json_to_string(obj, **kwargs))


def json_to_string(obj: Any, **kwargs: Any) -> Text:
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def write_to_file(filename: Text, text: Text) -> None:
    """Write a text to a file."""

    with io.open(filename, 'w', encoding="utf-8") as f:
        f.write(str(text))


def read_file(filename: Text, encoding: Text = "utf-8-sig") -> Any:
    """Read text from a file."""
    with io.open(filename, encoding=encoding) as f:
        return f.read()


def read_ssr_file(filename: Text) -> Any:
    content = read_file(filename)

    lines = content.split('\n')
    return [line for line in lines if str(line).startswith('ssr:')]


def decode_base64(context):
    text = context.replace("-", "+").replace("_", "/")
    text = bytes(text, encoding="utf-8")
    missing_padding = 4 - len(text) % 4

    if missing_padding:
        text += b'=' * missing_padding
        try:
            return str(base64.decodebytes(text), encoding="utf-8")
        except:
            return ""


config_keys = {"remarks", "server", "server_port", "method", "obfs", "obfsparam", "remarks_base64", "password",
               "tcp_over_udp", "udp_over_tcp", "protocol", "obfs_udp", "enable", "group"}


def lines2cfgs(lines: List):
    configs = []
    for line in lines:
        cfg = collections.defaultdict(str)
        format_ssr_url = line[6:]
        # print(format_ssr_url)
        server, server_port, protocol, method, obfs, other = decode_base64(format_ssr_url).split(":")
        password_base64, param_base64 = other.split("/?")
        password = decode_base64(password_base64)
        params = param_base64.split("&")

        for param in params:
            k, v = param.split("=", 1)
            if v:
                v = decode_base64(v)
            cfg[k] = v

        remarks_base64 = str(base64.b64encode(cfg['remarks'].encode('utf-8')), "utf-8")

        cfg.update({
            'server': server,
            'server_port': int(server_port),
            'protocol': protocol,
            'method': method,
            'obfs': obfs,
            'password': password,
            'remarks_base64': remarks_base64,
            'protocolparam': cfg.get('protoparam', ''),
            'enable': True,
        })
        configs.append(cfg)

    return configs


def main():
    lines = read_ssr_file(FLAGS.path)
    cfgs = lines2cfgs(lines)
    ssr_template.update({'configs': cfgs})
    write_json_to_file(FLAGS.output, ssr_template)


if __name__ == '__main__':
    main()
