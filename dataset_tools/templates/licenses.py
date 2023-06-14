class License:
    class Custom:
        def __new__(cls, url: str):
            self = super().__new__(cls)
            self.name = "custom"
            self.url = url
            return self

    class CC0_1_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC0 1.0"
            self.url = "https://creativecommons.org/publicdomain/zero/1.0/legalcode"
            return self

    class CC_BY_NC_SA_3_0_US:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY-NC-SA 3.0 US"
            self.url = "https://creativecommons.org/licenses/by-nc-sa/3.0/us/"
            return self

    class CC_BY_SA_4_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY-SA 4.0"
            self.url = "https://creativecommons.org/licenses/by-sa/4.0/legalcode"
            return self

    class CC_BY_NC_ND_4_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY-NC-ND 4.0"
            self.url = "https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode"
            return self

    class CC_BY_NC_SA_4_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY-NC-SA 4.0"
            self.url = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
            return self

    class CC_BY_4_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY 4.0"
            self.url = "https://creativecommons.org/licenses/by/4.0/legalcode"
            return self

    class BY_NC_SA_4_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY-NC-SA 4.0"
            self.url = "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode"
            return self

    class ODbL_1_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "Open Data Commons Open Database License v1.0"
            self.url = "https://opendatacommons.org/licenses/odbl/1-0/"
            return self

    class DbCL_1_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "Open Data Commons Database Contents License v1.0"
            self.url = "https://opendatacommons.org/licenses/dbcl/1-0/"
            return self