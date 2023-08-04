class License:
    class Custom:
        def __new__(cls, url: str = None, redistributable=True):
            self = super().__new__(cls)
            self.name = "custom"
            self.url = url
            self.redistributable = redistributable
            return self

    class Unknown:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "unknown"
            self.url = None
            self.redistributable = False
            return self

    class CC0_1_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC0 1.0"
            self.url = "https://creativecommons.org/publicdomain/zero/1.0/"
            self.redistributable = True
            return self

    class CC_BY_NC_2_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY-NC 2.0"
            self.url = "https://creativecommons.org/licenses/by-nc/2.0/"
            self.redistributable = True
            return self

    class CC_BY_NC_SA_3_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY-NC-SA 3.0 US"
            self.url = "https://creativecommons.org/licenses/by-nc-sa/3.0/"
            self.redistributable = True
            return self

    class CC_BY_NC_SA_3_0_IGO:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY-NC-SA 3.0 IGO"
            self.url = "https://creativecommons.org/licenses/by-nc-sa/3.0/igo/"
            self.redistributable = True
            return self

    class CC_BY_NC_SA_3_0_US:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY-NC-SA 3.0 US"
            self.url = "https://creativecommons.org/licenses/by-nc-sa/3.0/us/"
            self.redistributable = True
            return self

    class CC_BY_SA_4_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY-SA 4.0"
            self.url = "https://creativecommons.org/licenses/by-sa/4.0/legalcode"
            self.redistributable = True
            return self

    class CC_BY_NC_ND_4_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY-NC-ND 4.0"
            self.url = "https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode"
            self.redistributable = True
            return self

    class CC_BY_NC_SA_4_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY-NC-SA 4.0"
            self.url = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
            self.redistributable = True
            return self

    class CC_BY_4_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY 4.0"
            self.url = "https://creativecommons.org/licenses/by/4.0/legalcode"
            self.redistributable = True
            return self

    class CC_BY_NC_4_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY-NC 4.0"
            self.url = "https://creativecommons.org/licenses/by-nc/4.0/legalcode"
            self.redistributable = True
            return self

    class BY_NC_SA_4_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CC BY-NC-SA 4.0"
            self.url = "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode"
            self.redistributable = True
            return self

    class ODbL_1_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "ODbL v1.0"
            self.url = "https://opendatacommons.org/licenses/odbl/1-0/"
            self.redistributable = True
            return self

    class DbCL_1_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "DbCL v1.0"
            self.url = "https://opendatacommons.org/licenses/dbcl/1-0/"
            self.redistributable = True
            return self

    class MIT:
        def __new__(cls, url: str = None):
            self = super().__new__(cls)
            self.name = "MIT"
            self.url = url or "https://spdx.org/licenses/MIT.html"
            self.redistributable = True
            return self

    class Apache_2_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "Apache 2.0"
            self.url = "https://www.apache.org/licenses/LICENSE-2.0"
            self.redistributable = True
            return self

    class GNU_GPL_v3:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "GNU GPL 3.0"
            self.url = "https://www.gnu.org/licenses/gpl-3.0.en.html"
            self.redistributable = True
            return self

    class CDLA_Permissive_1_0:
        def __new__(cls):
            self = super().__new__(cls)
            self.name = "CDLA Permissive 1.0"
            self.url = "https://cdla.dev/permissive-1-0/"
            self.redistributable = True
            return self
