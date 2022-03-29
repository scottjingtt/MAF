import torch
import torch.nn as nn

class DomainCV():
    def __init__(self, feature_num):
        self.SourceVar = torch.zeros(feature_num, feature_num).cuda()
        self.SourceAve = torch.zeros(1,feature_num).cuda()
        self.SourceAmount = 0
    def update_CV(self, features):
        N = features.size(0)
        A = features.size(1)
        weight = N / (N+self.SourceAmount)
        # Mean
        batch_ave = features.sum(0)/N
        self.SourceAve = (1-weight)*self.SourceAve + weight*batch_ave
        # Covariance
        batch_diff = features - batch_ave.expand(N,A)
        batch_var = torch.matmul(batch_diff.t(), batch_diff) / N
        addition_CV = weight*(1-weight)*torch.matmul((self.SourceAve-batch_ave).view(A,1),
                                                  (self.SourceAve-batch_ave).view(1,A))



        self.SourceVar = (1-weight)*self.SourceVar + weight*batch_var + addition_CV

        self.SourceAmount += N

class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        # source domain class co-variance
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num)#.cuda()
        # source domain class centers
        self.Ave = torch.zeros(class_num, feature_num)#.cuda()
        self.Amount = torch.zeros(class_num)#.cuda()

        # target domain class centers
        self.tAve = torch.zeros(class_num, feature_num)#.cuda()
        self.tAmount = torch.zeros(class_num)#.cuda()

    def update_CV(self, features, labels):
        features = features.cpu()
        labels = labels.cpu()
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C)#.cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.cpu().view(C, 1, 1).expand(C, A, A)
        )

        weight_CV = weight_CV#.cpu()
        weight_CV[weight_CV != weight_CV] = 0
        weight_CV = weight_CV#.cuda()

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.cpu().view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave.cpu() - ave_CxA).view(C, A, 1),
                (self.Ave.cpu() - ave_CxA).view(C, 1, A)
            )
        )

        self.CoVariance = (self.CoVariance.cpu().mul(1 - weight_CV) + var_temp
                      .mul(weight_CV)).detach() + additional_CV.detach()#).cuda()

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()#).cuda()

        self.Amount += onehot.sum(0)#.cuda()

    def update_tAve(self, tfeatures, tlabels):
        tfeatures = tfeatures.cpu()
        tlabels = tlabels.cpu()
        N = tfeatures.size(0)
        C = self.class_num
        A = tfeatures.size(1)
        NxCxFeatures = tfeatures.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C)#.cuda()
        onehot.scatter_(1, tlabels.view(-1, 1), 1)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)
        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1
        ave_CxA = features_by_sort.sum(0) / Amount_CxA
        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)
        weight_AV = sum_weight_AV.div(sum_weight_AV + self.tAmount.view(C, 1).expand(C, A))
        weight_AV[weight_AV != weight_AV] = 0
        self.tAve = (self.tAve.mul(1-weight_AV) + ave_CxA.mul(weight_AV)).detach()#).cuda()
        self.tAmount += onehot.sum(0)#).cuda()



class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num)
        # Mask
        # self.domain = DomainCV(feature_num)


        self.class_num = class_num

        self.cross_entropy = nn.CrossEntropyLoss()

    def isda_aug(self, fc, features, y, labels, cv_matrix, cls, ratio_alpha, ratio_beta):

        N = features.size(0) # number of target samples
        C = self.class_num
        A = features.size(1)

        if cls == 'Ct':
            weight_m = list(fc.parameters())[0][C:,:]  # Ct weights, C x A
            bias_m = list(fc.parameters())[1][C:]
        elif cls == 'Cs':
            weight_m = list(fc.parameters())[0][:C, :]
            bias_m = list(fc.parameters())[1][:C]
        # print("fc weights shape: ", weight_m.shape, "fc bias shape: ", list(fc.parameters())[1].shape, "y pred: ", y.shape, " labels: ", labels.shape, "cv max: ", cv_matrix.shape)

        # 1. CCA - Cross-domain Continuity Augmentation
        # print(labels)
        y_true = labels # torch.arange(self.class_num).type(torch.LongTensor).cuda()
        eta_s = self.estimator.Ave[y_true, :].cuda() # Take each category's prototypes (eta) from class-0 to class-C
        eta_t = self.estimator.tAve[y_true, :].cuda()

        aug_feats = (1 - ratio_beta) * eta_s + ratio_beta * eta_t
        # y_pred = torch.mm(weight_m, aug_feats) + fc.parameters()[1]
        # y_pred = fc(aug_feats)
        # print("y pred shape: ", y_pred.shape)
        if cls == 'Cs':
            y_pred = fc(aug_feats)[:, :C]
            # return y_pred[:, :C]
        elif cls == 'Ct':
            y_pred = fc(aug_feats)[:, C:]
            # return y_pred[:, C:]
        else:
            raise Exception("Cs/Ct not defined!!!")

        # 2. SSA - Source-supervised Semantic Augmentation
        NxW_ij = weight_m.expand(N, C, A) # C x A --> N x C x A

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A)) # For each (xt_i, yt_i), gather the "yt_i" weights of NxW_ij

        CV_temp = cv_matrix[labels]  # Take the covariance of each yt_i from source covariance [C, A, A]
        CV_temp = CV_temp.cuda()

        sigma2 = ratio_alpha * \
                 torch.bmm(torch.bmm(NxW_ij - NxW_kj,
                                     CV_temp),
                           (NxW_ij - NxW_kj).permute(0, 2, 1))

        A = 0.5 * sigma2.mul(torch.eye(C).cuda()
                            .expand(N, C, C)).sum(2).view(N, C)

        aug_result = y_pred + A

        return aug_result

    def forward(self, model, xs, xt, s_features, t_features, target_s,target_t, yt_pred, cls, ratio_alpha, ratio_beta):
        fc = model.module.fc
        self.estimator.update_CV(s_features.detach(), target_s.detach())
        self.estimator.update_tAve(t_features.detach(), target_t.detach())
        # print(type(self.estimator.CoVariance))
        # print(self.estimator.CoVariance.shape)
        isda_aug_y = self.isda_aug(fc, t_features, yt_pred, target_t, self.estimator.CoVariance.detach(), cls, ratio_alpha, ratio_beta) #self.estimator.CoVariance.detach()

        loss = self.cross_entropy(isda_aug_y, target_t)

        return loss #, self.estimator