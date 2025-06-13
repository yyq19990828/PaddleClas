# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import logging
import os
import sys

import paddle.distributed as dist

_logger = None


class LoggerHook(object):
    """
    logs will print multi-times when calling Fleet API.
    Commonly, only need to display single log at rank0 and ignore the others.
    """
    block = False

    def __init__(self, log):
        self.log = log

    def __call__(self, *args, **kwargs):
        # 找到调用栈中第一个logger文件以外的调用
        kwargs['stacklevel'] =3
        if not self.block:
            self.log(*args, **kwargs)

# 自定义日志记录器的 Filter，用于添加相对路径
class RelativePathFilter(logging.Filter):
    def filter(self, record):
        record.relativepath = os.path.relpath(record.pathname, start=os.getcwd())
        return True

class ColorFormatter(logging.Formatter):
    """自定义日志格式化器，支持颜色和加粗"""
    COLOR_CODES = {
        'DEBUG': '\033[1;94m',  # 加粗蓝色
        'INFO': '\033[1;92m',   # 加粗绿色
        'WARNING': '\033[1;93m',  # 加粗黄色
        'ERROR': '\033[1;91m',  # 加粗红色
        'CRITICAL': '\033[1;101m',  # 加粗红色背景
    }
    RESET_CODE = '\033[0m'  # 重置颜色

    def format(self, record):
        # 获取日志级别对应的颜色并加粗
        color = self.COLOR_CODES.get(record.levelname, self.RESET_CODE)
        # 创建记录的副本，避免修改原始记录（可能被其他处理器使用）
        record_copy = logging.makeLogRecord(record.__dict__)
        # 修改消息头的格式，带颜色加粗
        record_copy.levelname = f"{color}{record.levelname}{self.RESET_CODE}"
        record_copy.name = f"\033[1m{record.name}{self.RESET_CODE}"
        record_copy.relativepath = f"\033[1m{record.relativepath}{self.RESET_CODE}"
        record_copy.funcName = f"\033[1m{record.funcName}{self.RESET_CODE}"
        # 保持 lineno 为整数类型，避免格式化错误
        return super().format(record_copy)

def init_logger(name='PaddleClas',
                log_file=None,
                log_level=logging.INFO,
                log_ranks="0"):
    """通过名称初始化并获取一个日志记录器。
    如果日志记录器尚未初始化，此方法将通过添加一个或两个处理器来初始化日志记录器，
    否则将直接返回已初始化的日志记录器。在初始化期间，总是会添加一个 StreamHandler。
    如果指定了 `log_file`，还会添加一个 FileHandler。
    
    参数:
        name (str): 日志记录器名称。
        log_file (str | None): 日志文件名。如果指定，将为日志记录器添加一个 FileHandler。
        log_level (int): 日志记录器级别。注意，仅 rank 0 的进程会受到影响，
            其他进程将日志级别设置为 "Error"，因此大多数情况下是静默的。
        log_ranks (str): 需要记录日志的 GPU ID，多个 ID 用 "," 分隔，默认为 "0"。
    
    返回:
        logging.Logger: 所需的日志记录器。
    """
    global _logger

    #  solve mutiple init issue when using paddleclas.py and engin.engin
    init_flag = False
    if _logger is None:
        _logger = logging.getLogger(name)
        init_flag = True

    # 为终端输出创建彩色格式化器（去除实时时间显示）
    color_formatter = ColorFormatter(
        '%(name)s %(levelname)s [%(funcName)s %(relativepath)s:%(lineno)d]: %(message)s'
    )
    
    # 为文件日志创建普通格式化器（不带颜色代码）
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s %(funcName)s %(relativepath)s:%(lineno)d]: %(message)s',
        datefmt="%Y/%m/%d %H:%M:%S"
    )

    # 配置终端处理器
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.addFilter(RelativePathFilter()) # 添加过滤器
    stream_handler.setFormatter(color_formatter)  # 使用彩色格式化器
    stream_handler._name = 'stream_handler'

    # add stream_handler when _logger dose not contain stream_handler
    for i, h in enumerate(_logger.handlers):
        if h.get_name() == stream_handler.get_name():
            break
        if i == len(_logger.handlers) - 1:
            _logger.addHandler(stream_handler)
    if init_flag:
        _logger.addHandler(stream_handler)

    # 配置文件处理器
    if log_file is not None and dist.get_rank() == 0:
        log_file_folder = os.path.split(log_file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        file_handler = logging.FileHandler(log_file, 'a')
        file_handler.addFilter(RelativePathFilter())  # 添加相同的过滤器
        file_handler.setFormatter(file_formatter)  # 明确使用普通格式化器，不带颜色
        file_handler._name = 'file_handler'

        # add file_handler when _logger dose not contain same file_handler
        for i, h in enumerate(_logger.handlers):
            if h.get_name() == file_handler.get_name() and \
                    h.baseFilename == file_handler.baseFilename:
                # 确保已有的file_handler也使用正确的formatter
                h.setFormatter(file_formatter)
                break
            if i == len(_logger.handlers) - 1:
                _logger.addHandler(file_handler)

    if isinstance(log_ranks, str):
        log_ranks = [int(i) for i in log_ranks.split(',')]
    elif isinstance(log_ranks, int):
        log_ranks = [log_ranks]
    if dist.get_rank() in log_ranks:
        _logger.setLevel(log_level)
        LoggerHook.block = False
    else:
        _logger.setLevel(logging.ERROR)
        LoggerHook.block = True
    _logger.propagate = False


@LoggerHook
def info(fmt, *args, **kwargs):
    _logger.info(fmt, *args, **kwargs)


@LoggerHook
def debug(fmt, *args, **kwargs):
    _logger.debug(fmt, *args, **kwargs)


@LoggerHook
def warning(fmt, *args, **kwargs):
    _logger.warning(fmt, *args, **kwargs)


@LoggerHook
def error(fmt, *args, **kwargs):
    _logger.error(fmt, *args, **kwargs)


def scaler(name, value, step, writer):
    """
    This function will draw a scalar curve generated by the visualdl.
    Usage: Install visualdl: pip3 install visualdl==2.0.0b4
           and then:
           visualdl --logdir ./scalar --host 0.0.0.0 --port 8830 
           to preview loss corve in real time.
    """
    if writer is None:
        return
    writer.add_scalar(tag=name, step=step, value=value)


def advertise():
    """
    Show the advertising message like the following:

    ===========================================================
    ==        PaddleClas is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleClas      ==
    ===========================================================

    """
    copyright = "PaddleClas is powered by PaddlePaddle !"
    ad = "For more info please go to the following website."
    website = "https://github.com/PaddlePaddle/PaddleClas"
    AD_LEN = 6 + len(max([copyright, ad, website], key=len))

    info("\n{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}\n".format(
        "=" * (AD_LEN + 4),
        "=={}==".format(copyright.center(AD_LEN)),
        "=" * (AD_LEN + 4),
        "=={}==".format(' ' * AD_LEN),
        "=={}==".format(ad.center(AD_LEN)),
        "=={}==".format(' ' * AD_LEN),
        "=={}==".format(website.center(AD_LEN)),
        "=" * (AD_LEN + 4), ))
