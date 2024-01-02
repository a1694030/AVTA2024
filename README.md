# icgsp2024
1. Introduction
1. 背景介绍
近年来，随着数字经济的发展，各类app平台会向现有客户推荐交叉销售的金融产品, strives to broaden financial inclusion。为此他们使用不同类型的数据，一方面希望能够直接定位到潜在客户再进行精准营销，另一方面希望降低金融信贷的不良率。此时,除了不断优化信用卡营销策略外, 提高金融业务的转化率和节省推广成本，各大涉及金融业务的公司也十分注重金融借贷风险。同时，传统的银行业也从人工分析模式加速转型为数字金融, by providing a positive and safe borrowing experience。特别是5G网络的兴起，移动端消费的增加，移动金融成为金融业发展的新方向。传统的信用卡发放和潜在用户分析效率低下，预测偏差大，因此，如何利用移动通讯积累的海量数据与深度学习技术以最小的营销资源通过量化分析获取到最大量的信用卡潜在新用户，并作出相应的风险分析是银行业迫在眉睫的问题。


2. 用处
由于移动消费与移动通讯流量之间具有相关性，因此可以间接的基于每月用户使用5G套餐数据，分析用户消费的行为特征，从移动端数据，包括：基础信息、消费行为、位置信息、移动轨迹信息、其他信息等维度描绘出用户画像，构建信用卡潜客(the unbanked population)识别模型，识别出信用卡高需求低风险潜客群体进行营销。

现在的问题，传统方法对于潜客识别的研究不多，且精度不够

传统统计学方法在精准识别5G移动用户是否是信用卡需求潜在用户中力不从心，在风险评估方面不能满足风险管控需要.

([Predicting New Customers’ Risk Type in the Credit Card Market]


)

客户识别分析具有局限性：随着信息技术的发展，咨询传播范围广，加上网络诈骗猖獗，金融机构高质量获取客户变得困难，银行在客户获取的方式上，还是使用传统投放广告、传单等方法，高质量客户的开发效率不高。
信用卡业务风险多：由于我国银行信用卡的使用范围较广，发放数量规模大，所以存在一定程度上的风险。为了降低风险，需要银行的管理者设置风险监管机构，防范信用卡的业务风险。这样就导致了信用卡办理手续繁杂，与潜在的高质量客户匹配灵活性差。
宣传营销成本高：信用卡的营销方式主要有联名卡合作、广告投放、信用卡推广网站和社区宣传等营销方式。这些营销方式需要投入大量的资金和资源，还需要专门制定不同的投放策略，不但会额外增加成本，而且潜在客户的质量无法得到相应的保证，难以获得有效的回报。
精准定位技术不高：只有优质的金融产品才能为金融机构带来丰厚的回报。但随着外资机构、新兴金融企业、私有制银行的涌入与崛起，金融机构对优质客户的竞争也愈发激烈。当前，如何打造满足用户需求的金融产品，是行业内各家金融机构都在重点探索的关键领域。基于大数据技术，通过开发新的金融产品或营销方式，有望给金融机构带来业务上的新突破。






机器学习或者单纯的深度学习的不足
Common machine learning methods, such as decision trees and multi-layer perceptrons, are prone to overfitting in such problems and have difficulty achieving good performance.

([Machine Learning Mini Batch K-means and Business Intelligence Utilization for Credit Card Customer Segmentation]

[A credit card usage behaviour analysis framework - a data mining approach]
)


3. 现有数据集不足，数据集来源单一或者没有精确介绍来源并公开
此外，虽然各个银行已经单纯从他们的业务数据确定了一组有潜在资格使用信用卡的客户。但是，信用卡业务竞争激烈，各大银行仅仅通过本身获取的数据来推荐更高意向的潜在信用卡客户不够精确。

([Targeting Customers with Data Mining Techniques: Classification]机场数据
[A Hybrid Data Mining Approach for Credit Card Usage Behavior Analysis]台湾数据
[An Xgboost based system for financial fraud detection] Kaggle数据
)


我们的数据集：
我们的数据集全部来源于 第三届中国移动“梧桐杯” 大数据创新大赛暨大数据创客马拉松大赛的 数据应用赛道 的公共数据集。
