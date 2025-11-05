import requests
import re
def linked_post_fun(var):
    url = "https://www.linkedin.com/voyager/api/graphql?action=execute&queryId=voyagerContentcreationDashShares.12edc89bf35c59e5b05ae2eb5f51fd28"

    headers = {
        "accept": "application/vnd.linkedin.normalized+json+2.1",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json; charset=UTF-8",
        "csrf-token": "ajax:4133810535057557344",
        "priority": "u=1, i",
        "sec-ch-prefers-color-scheme": "dark",
        "sec-ch-ua": "\"Google Chrome\";v=\"141\", \"Not?A_Brand\";v=\"8\", \"Chromium\";v=\"141\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-li-lang": "en_US",
        "x-li-page-instance": "urn:li:page:d_flagship3_company_admin;5df9af38-9909-4f3a-8d73-3bba4234a75c",
        "x-li-pem-metadata": "Voyager - Sharing - CreateShare=sharing-create-content,Voyager - Organization - Admin=organization-create-post-as-page",
        "x-li-track": "{\"clientVersion\":\"1.13.39719\",\"mpVersion\":\"1.13.39719\",\"osName\":\"web\",\"timezoneOffset\":5.5,\"timezone\":\"Asia/Calcutta\",\"deviceFormFactor\":\"DESKTOP\",\"mpName\":\"voyager-web\",\"displayDensity\":1.25,\"displayWidth\":1920,\"displayHeight\":1080}",
        "x-restli-protocol-version": "2.0.0",
        "cookie": "bcookie=\"v=2&877cca2b-5156-476b-84ec-1966a9769d42\"; bscookie=\"v=1&2022030209204642c75bd9-7695-45c6-859e-4e374ff3466eAQE3Gq3PRO2CAlZhNQXKBQKU4_O2Tqov\"; li_sugr=655d2920-bfb5-4768-9c63-4ec591c48780; li_theme=light; li_theme_set=app; dfpfpt=af7e00b5c823464f852d26f24130cbda; timezone=Asia/Calcutta; _guid=d4458fe3-0012-4716-b1ba-4c3d6732718f; aam_uuid=42678121612710579671122974409925658551; li_rm=AQE113pjPWcZFAAAAZlIYHsqLeBc-s9NTQEvW93KPvZ_8iU3tuzWXp8nvcHGdHLLj6z0bNlJHlnX19bZPWF4O-LCDOK5Ombx5M_LEeT8qtOZsA3ap-wCzDhK1_RArOv6kOV357L3YpAo2SepJAv_tNo6tnWNyBmgXNFzSZ4vTItRIrKgSI8p-IgMDnZhooKNa0sx5CX9tOiFtKZbbTNavEXQdAX3IpHUBj-E7Int6-tiCAecgURuC4HgwHKzEJbaJg_geHq9UyTJ7DsguroGUyy_8p9NpVwS243VpJ3H41tek3iifEHEFBmahRUnXfaoYlI7bdP5sU5xlT6RzbPlCg; visit=v=1&M; g_state={\"i_l\":0}; JSESSIONID=\"ajax:4133810535057557344\"; s_ips=695; _gcl_au=1.1.208960101.1757357595.751433026.1758524184.1758524744; sdui_ver=sdui-flagship:0.1.14359+SduiFlagship0; AnalyticsSyncHistory=AQICBWfDgOxPtgAAAZnOTXf8ZoxPmyZ1MhdYQLsOsoYlgvPCTOh0bSQIbkNhWPN_oPwJU9yCtcEQe-BQs1RneA; lms_ads=AQHV8e8kKFx4EQAAAZnOTXm2PWD4X7nAHJKP73mjHwR2yfEVBNooz9bIDsC10kqSod3hHqgQL1x58-s2Zvz8k4I9_HjIK6ez; lms_analytics=AQHV8e8kKFx4EQAAAZnOTXm2PWD4X7nAHJKP73mjHwR2yfEVBNooz9bIDsC10kqSod3hHqgQL1x58-s2Zvz8k4I9_HjIK6ez; lang=v=2&lang=en-us; AMCVS_14215E3D5995C57C0A495C55%40AdobeOrg=1; AMCV_14215E3D5995C57C0A495C55%40AdobeOrg=-637568504%7CMCIDTS%7C20372%7CMCMID%7C43204998124350827241106395094946935932%7CMCAAMLH-1760778649%7C12%7CMCAAMB-1760778649%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1760181049s%7CNONE%7CMCCIDH%7C1112151125%7CvVersion%7C5.1.1; fptctx2=taBcrIH61PuCVH7eNCyH0FWPWMZs3CpAZMKmhMiLe%252bGb4Co%252fMwBV5qPD%252f23LnQK6qr0U6BeHV%252fId7%252b7ScfCTqd7Vj4es4s%252ffsX1SWhE14JlQ7iuF0uzSsjUKj4eto58qeIAjNq4MtER6yomSrigBVtnyy%252fn7r6sHQPSeWN4nkWfIcNPCoKAy9vAbH9UrcBpzu6qfQusoyHtnT3SrlzoAJyUV%252f01TsSjzi3jdT7Kt9UC3ap00DNdoxQ6VXi2p%252fOqNRBxBpV2qsJ16epSK69MIOubO83l%252fFnaDPYNM6ZjTrJ98iE3OYKfcQNRwXuKTDsyVRqzN7kz08e5SFOnykLxdicgG1biQVXuuKaw%252bIIe4Pc8%253d; li_at=AQEDATXb6KwDyvFrAAABmdK9thQAAAGZ9so6FE0A0tT67J6vX5gCtdlqlNbv4hoQCsCdWGWgVvuGnfFgCIPuopf_YznnC93EPE-EJFwH1llyE7p-bIAdKcZWPRUEiKQI0xfsM6ZelSIAbInJjGk8GuHp; liap=true; bitmovin_analytics_uuid=11b8d20c-78f7-4657-9d4b-16bdb85c78d4; at_check=true; mbox=PC#859257aa1dff475eb1ac876f95dd0c5b.41_0#1775730681|session#2ef9bb349daf49d5a753f7d6a5a7dde2#1760180541; _uetsid=6d96cbb0a68d11f0863273861a1c84ce; _uetvid=2825ada08ce511f083adb7886f3bfeab; s_cc=true; UserMatchHistory=AQKCQo9RfFgkEgAAAZnS06nuNj0NBmN1D4PlWJNZGBhAq91Tdte33QhHlC7IWI0-Mj-FJymysGWn_45swBrBCl1XWOQM_slVMp3IZuqy0TpqHwF1IxeO9FeYuM_Lc09LIj99dhOrLNS_HRs1lFXCT57DTuF8vt7wT60vMVs8OJy6hFq8sZeaAaAVYiKtbt2-QNrNkZ7nDgn6dUcRWhIYFP1wUfqV7u9Wwx9TRJH28hEp1w7pWMAJ-qXstK0Qo6gqJxQTzsVV7zVFx4B4IIc7-R34wbOQozqtZwJPTQq7gELI-fWFocQ60ExPUHJYr3JLdmtSTMB1fRAjDgv-B608eknXk3nWj_5_y0wZhFEw-0f6qjSwOQ; gpv_pn=www.linkedin.com%2Fcompany%2Fid-redacted%2Fadmin%2Fpage-posts%2Fpublished%2F; s_plt=7.66; s_pltp=www.linkedin.com%2Fcompany%2Fid-redacted%2Fadmin%2Fpage-posts%2Fpublished%2F; lidc=\"b=VB96:s=V:r=V:a=V:p=V:g=4093:u=126:x=1:i=1760178715:t=1760263593:v=2:sig=AQGr0irLbBb9dE_NCuVJdzAXV8AiUCqZ\"; s_tp=2214; s_ppv=www.linkedin.com%2Fcompany%2Fid-redacted%2Fadmin%2Fpage-posts%2Fpublished%2F%2C31%2C31%2C695%2C1%2C3; s_tslv=1760178846518; s_sq=lnkdprod%3D%2526c.%2526a.%2526activitymap.%2526page%253Dwww.linkedin.com%25252Fcompany%25252Fid-redacted%25252Fadmin%25252Fpage-posts%25252Fpublished%25252F%2526link%253DPost%2526region%253Dember238%2526pageIDType%253D1%2526.activitymap%2526.a%2526.c%2526pid%253Dwww.linkedin.com%25252Fcompany%25252Fid-redacted%25252Fadmin%25252Fpage-posts%25252Fpublished%25252F%2526pidt%253D1%2526oid%253DPost%2526oidt%253D3%2526ot%253DSUBMIT",
        "Referer": "https://www.linkedin.com/company/109142404/admin/page-posts/published/?share=true"
    }

    payload = {
        "variables": {
            "post": {
                "allowedCommentersScope": "ALL",
                "intendedShareLifeCycleState": "PUBLISHED",
                "origin": "ORGANIZATION",
                "visibilityDataUnion": {
                    "visibilityType": "ANYONE"
                },
                "commentary": {
                    "text": var
                },
                "nonMemberActorUrn": "urn:li:fsd_company:109142404"
            }
        },
        "queryId": "voyagerContentcreationDashShares.12edc89bf35c59e5b05ae2eb5f51fd28",
        "includeWebMetadata": True
    }
    clean_text = re.sub(r"\*", "",var)
    response = requests.post(url, headers=headers, json=payload)
    print(f"its running with str {var}")

    #print("Status code:", response.status_code)
    #print("Response:", response.text)
    return response.status_code


# if __name__ == "__main__":

#     resp = linked_post_fun(var="its running with str **Job Title: Data Scientist** **Company: Laxmi Chit Fund** **Location: [Location]** **About Us:** Laxmi Chit Fund is a premier financial service provider specializing in innovative financial solutions. We leverage technology and data to enhance our business operations and offer superior service to our clients. We are seeking a highly skilled and motivated Data Scientist to join our team and contribute to our data-driven decision-making processes. **Job Description:** As a Data Scientist at Laxmi Chit Fund, you will play a crucial role in harnessing the power of data to drive business strategies and optimize operations. You will be responsible for developing and implementing advanced analytics, machine learning, and deep learning models that improve business outcomes. You will work closely with cross-functional teams to identify opportunities for leveraging data rning, and statistical modeling to interpret and analyze complex data sets. - Implement and manage the MLOps framework to streamline the process of deploying, testing, monitoring, and maintaining machine learning models. - Use Python and other programming languages to cleanse, analyze, and visualize data, ensuring high-quality data transformations. - CommunicatPower BI. - Experience with cloud-based services such as AWS, Google Cloud, or Azure. - Understanding of big data technologies such as Hadoop or Spark. **What We Offer:** - Competitive salary and benefits package. - Opportunity to work in a dynamic and innovative environment. - Professional growth and development opportunities. - Collaborative and inclusive company culture. If you are passionate about data science and keen on applying your skills to make impactful business decisions, we encourage you to apply for this exciting opportunity at Laxmi Chit Fund. **How to Apply:** Interested candidates can send their resume and a cover letter to [Email Address] with the subject line "Application for Data Scientist - Laxmi Chit Fund." Applications will be reviewed o")

