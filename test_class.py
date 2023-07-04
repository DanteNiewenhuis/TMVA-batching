# %%


class A:
    def foo_a(self):
        print("in foo_a")


# %%

a = A()
# %%


def foo_b():
    print("in foo_b")


setattr(a, "foo_b", foo_b)

# %%

a.foo_b()
