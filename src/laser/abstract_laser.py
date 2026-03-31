import torch


class AbstractLaser:

    # Helper functions for matrix update
    @staticmethod
    def get_parameter(model, name):
        for n, p in model.named_parameters():
            if n == name:
                return p
        raise LookupError(name)

    @staticmethod
    def update_model(model, name, params):
        with torch.no_grad():
            print('params')
            print(params)
            print('type(params)')
            print(type(params))
            AbstractLaser.get_parameter(model, name)[...] = params
    # 从这里可以看出，通过比对model.named_parameters()中自带的name(n)和外部输入的name，来进行赋值（请注意，这里的params就是后面用来赋予修改后的值的）


    @staticmethod
    def get_edited_model(model, lname, lnum, rate, intervention="rank-reduction", logger=None, in_place=True):
        raise NotImplementedError()
