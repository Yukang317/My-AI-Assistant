import traceback

def inner():
    try:
      1 / 0                         # 真正炸点
    except ZeroDivisionError as e:
      print("=== raise e ===")
      try:
          raise e                   # 重置 traceback
      except:
          traceback.print_exc()         # 打印异常堆栈信息

      print("\n=== 裸 raise ===")     # 打印异常堆栈信息
      raise                          # 保留原始 traceback，再次抛出异常

def middle():
  inner()

def outer():
  try:
      middle()
  except:
      traceback.print_exc()

outer()