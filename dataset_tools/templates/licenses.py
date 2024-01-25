class License:
    class Custom:
        def __new__(cls, source_url: str = None, redistributable=True, **kwargs):
            self = super().__new__(cls)
            self.name = "custom"
            self.url = kwargs.get("url", source_url)
            self.redistributable = redistributable
            self.source_url = kwargs.get("url", source_url)
            return self

    class Unknown:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "unknown"
            self.url = None
            self.redistributable = False
            self.source_url = source_url
            return self

    class PubliclyAvailable:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "publicly available"
            self.url = None
            self.source_url = source_url
            self.redistributable = True
            return self

    class BSD_3_Clause:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "3-Clause BSD License"
            self.url = "https://opensource.org/license/bsd-3-clause/"
            self.redistributable = True
            self.source_url = source_url
            return self

    class CC0_1_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CC0 1.0"
            self.url = "https://creativecommons.org/publicdomain/zero/1.0/"
            self.redistributable = True
            self.source_url = source_url
            return self

    class CC_BY_NC_2_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CC BY-NC 2.0"
            self.url = "https://creativecommons.org/licenses/by-nc/2.0/"
            self.redistributable = True
            self.source_url = source_url
            return self

    class CC_BY_NC_SA_3_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CC BY-NC-SA 3.0 US"
            self.url = "https://creativecommons.org/licenses/by-nc-sa/3.0/"
            self.redistributable = True
            self.source_url = source_url
            return self

    class CC_BY_NC_SA_3_0_IGO:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CC BY-NC-SA 3.0 IGO"
            self.url = "https://creativecommons.org/licenses/by-nc-sa/3.0/igo/"
            self.redistributable = True
            self.source_url = source_url
            return self

    class CC_BY_NC_SA_3_0_US:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CC BY-NC-SA 3.0 US"
            self.url = "https://creativecommons.org/licenses/by-nc-sa/3.0/us/"
            self.redistributable = True
            self.source_url = source_url
            return self

    class CC_BY_SA_4_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CC BY-SA 4.0"
            self.url = "https://creativecommons.org/licenses/by-sa/4.0/legalcode"
            self.redistributable = True
            self.source_url = source_url
            return self

    class CC_BY_NC_ND_2_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CC BY-NC-ND 2.0"
            self.url = "https://creativecommons.org/licenses/by-nc-nd/2.0/"
            self.redistributable = True
            self.source_url = source_url
            return self

    class CC_BY_NC_ND_4_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CC BY-NC-ND 4.0"
            self.url = "https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode"
            self.redistributable = True
            self.source_url = source_url
            return self

    class CC_BY_NC_SA_4_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CC BY-NC-SA 4.0"
            self.url = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
            self.redistributable = True
            self.source_url = source_url
            return self

    class CC_BY_3_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CC BY 3.0"
            self.url = "https://creativecommons.org/licenses/by/3.0"
            self.redistributable = True
            self.source_url = source_url
            return self

    class CC_BY_4_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CC BY 4.0"
            self.url = "https://creativecommons.org/licenses/by/4.0"
            self.redistributable = True
            self.source_url = source_url
            return self

    class CC_BY_NC_3_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CC BY-NC 3.0"
            self.url = "https://creativecommons.org/licenses/by-nc/3.0/"
            self.redistributable = True
            self.source_url = source_url
            return self

    class CC_BY_NC_4_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CC BY-NC 4.0"
            self.url = "https://creativecommons.org/licenses/by-nc/4.0/legalcode"
            self.redistributable = True
            self.source_url = source_url
            return self

    class BY_NC_SA_4_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CC BY-NC-SA 4.0"
            self.url = "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode"
            self.redistributable = True
            self.source_url = source_url
            return self

    class ODbL_1_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "ODbL v1.0"
            self.url = "https://opendatacommons.org/licenses/odbl/1-0/"
            self.redistributable = True
            self.source_url = source_url
            return self

    class DbCL_1_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "DbCL v1.0"
            self.url = "https://opendatacommons.org/licenses/dbcl/1-0/"
            self.redistributable = True
            self.source_url = source_url
            return self

    class Etalab_2_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "Etalab Open License 2.0"
            self.url = "https://www.etalab.gouv.fr/wp-content/uploads/2018/11/open-licence.pdf"
            self.redistributable = True
            self.source_url = source_url
            return self

    class MIT:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "MIT"
            self.url = "https://spdx.org/licenses/MIT.html"
            self.redistributable = True
            self.source_url = source_url
            return self

    class NCGL_2_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "NCGL v2.0"
            self.url = "https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/"
            self.redistributable = True
            self.source_url = source_url
            return self

    class Apache_2_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "Apache 2.0"
            self.url = "https://www.apache.org/licenses/LICENSE-2.0"
            self.redistributable = True
            self.source_url = source_url
            return self

    class GNU_GPL_v3:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "GNU GPL 3.0"
            self.url = "https://www.gnu.org/licenses/gpl-3.0.en.html"
            self.redistributable = True
            self.source_url = source_url
            return self

    class GNU_AGPL_v3:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "GNU GPL 3.0"
            self.url = "https://www.gnu.org/licenses/agpl-3.0.html"
            self.redistributable = True
            self.source_url = source_url
            return self

    class GNU_GPL_v2:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "GNU GPL 2.0"
            self.url = "https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html"
            self.redistributable = True
            self.source_url = source_url
            return self

    class GNU_LGPL_v3:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "GNU LGPL 3.0"
            self.url = "https://www.gnu.org/licenses/lgpl-3.0.html"
            self.redistributable = True
            self.source_url = source_url
            return self

    class CDLA_Permissive_1_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CDLA Permissive 1.0"
            self.url = "https://cdla.dev/permissive-1-0/"
            self.redistributable = True
            self.source_url = source_url
            return self

    class CDLA_Permissive_2_0:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "CDLA Permissive 2.0"
            self.url = "https://cdla.dev/permissive-2-0/"
            self.redistributable = True
            self.source_url = source_url
            return self

    class OpenAccess:
        def __new__(cls, source_url: str = None):
            self = super().__new__(cls)
            self.name = "OpenAccess"
            self.url = "http://purl.org/eprint/accessRights/OpenAccess"
            self.redistributable = True
            self.source_url = source_url
            return self
