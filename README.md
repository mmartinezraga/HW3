# HW3
---
title: "HW3"
author: "M.Martinez-Raga"
date: "9/29/2020"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_knit$set(root.dir = "")
getwd()
load("acs2017_ny_data.RData")
acs2017_ny[1:10,1:7]
attach(acs2017_ny)
```

<p style="color:rgb(182,18,27);font-family:corbel">Mónica Martínez-Raga</p>
<p style="color:rgb(182,18,27);font-family:corbel">HW3- Fall 2020</p>
<p style="color:rgb(182,18,27);font-family:corbel">k-nn</p>

We build a k-nn algorithm to classify New Yorkers within boroughs. Below, the acs2017_ny was filtered for adults who live in the 5 boroughs.
```{r}
dat_NYC <- subset(acs2017_ny, (acs2017_ny$in_NYC == 1)&(acs2017_ny$AGE > 20) & (acs2017_ny$AGE < 66))
attach(dat_NYC)
borough_f <- factor((in_Bronx + 2*in_Manhattan + 3*in_StatenI + 4*in_Brooklyn + 5*in_Queens), levels=c(1,2,3,4,5),labels = c("Bronx","Manhattan","Staten Island","Brooklyn","Queens"))
```

Originally, the variables used to classify were income per person (INCTOT), as well as housing costs (OWNCOST + RENT). This resulted in about a 34-37% accuracy since average income and housing costs vary more within neighborhoods, not as much within boroughs. 

In order to increase accuracy, the k-nn algo will run with housing and commute data (TRANWORK, TRANTIME) variables. The idea behind this was to focus on attributes that depend more on the location and therefore vary less. Housing costs depends on market value of infrastructure and neighborhood, commute time depends on distance to economic centers and commute type, and type of work commute depends on how communicated the area is to public transport. I excluded income because after running housing, commute and income variables once, the accuracy decreased. I assume this is because income is not fixed for many people, but cost and commute tolerace tend to be. In other words, we may have more rogue higher earners living in places where their neighbors earn way less, however their housing and commute times are the same. 



First, I included a function that normalizes any variable in the dat_NYC data set. This is necessary in order to equalized the lengths of each vector.

```{r}
norm_varb <- function(X_in) {
  (X_in - min(X_in, na.rm = TRUE))/( max(X_in, na.rm = TRUE) - min(X_in, na.rm = TRUE) )
}
```

Next, we define the each variable and normalize using function. 

```{r}
OWNCOST <- dat_NYC$OWNCOST
RENT <- dat_NYC$RENT
is.na(OWNCOST) <- which(OWNCOST == 9999999)
is.na(RENT) <- which(RENT == 0)
housing_cost <- OWNCOST + RENT
norm_housing_cost <- norm_varb(housing_cost)
```

Since TRANWORK is integral, meaning that each number represents a type of commute, we change to numeric to make it compatible with the rest of the vectors. 
```{r}
TRANWORK <- dat_NYC$TRANWORK
TRANWORK <- as.numeric(TRANWORK)
is.na(TRANWORK) <- which(TRANWORK == 0)
norm_TRANWORK <- norm_varb(TRANWORK)
TRANTIME <- dat_NYC$TRANTIME
is.na(TRANTIME) <- which(TRANTIME == 0)
norm_TRANTIME <- norm_varb(TRANTIME)
```

Before we move on, we can check each variable within each borough to further strengthen our assumptions.

TRANWORK legend:
<p style="color:rgb(182,18,27);font-family:corbel">10-20 = Private vehicles</p>
<p style="color:rgb(182,18,27);font-family:corbel">10 = Auto, truck, or van</p>
<p style="color:rgb(182,18,27);font-family:corbel">30-36 = Public transport</p>
<p style="color:rgb(182,18,27);font-family:corbel">33 = Subway, or elevated</p>
<p style="color:rgb(182,18,27);font-family:corbel">40 = Bicycle</p>
<p style="color:rgb(182,18,27);font-family:corbel">50 = Walked only</p>
<p style="color:rgb(182,18,27);font-family:corbel">60 = Other</p>
<p style="color:rgb(182,18,27);font-family:corbel">70 = Worked at home</p>

For all boroughs except SI, the median is 33, meaning most people commute via subway. Within those 4 boroughs, only Manhattan's mean is above the median, suggesting that less people take private vehicles, and that maybe more people don't even need to take the metro since they are closer to their jobs. This makes sense because Manhattan is the biggest and best communicated economic center in NYC, and living is also more condensed than in other boroughs. Queens and the Bronx is where we would assume more people need private vehicles after SI because of limited of MTA reach (DE BLASIO!)

This allows us to assume that this data is good for classifying within boroughs given the variability, however the differences don't seem to be extremely significant.
```{r}
summary(TRANWORK[in_Manhattan == 1])
summary(TRANWORK[in_Bronx == 1])
summary(TRANWORK[in_Brooklyn == 1])
summary(TRANWORK[in_Queens == 1])
summary(TRANWORK[in_StatenI == 1])
```

Again we see a similar trend. Queens and the Bronx average the longest commute time to work because of lack of public transport infrastructure, less proximity to downtown Manhattan, and low housing density, while Manhattan remains the shortest commute.
```{r}
summary(TRANTIME[in_Manhattan == 1])
summary(TRANTIME[in_Bronx == 1])
summary(TRANTIME[in_Brooklyn == 1])
summary(TRANTIME[in_Queens == 1])
summary(TRANTIME[in_StatenI == 1])
```

We also see variability that matches our assumption that housing cost in Manhattan is the most expensive, the Bronx being the least. I'm surprised however to see that Queens is second most expensive. Maybe this is more intuitive if we classify by neighborhood. I can only assume Queens is beig enough to not only have expensive housing in Astoria and LIC but also closer to Long Island.
```{r}
summary(housing_cost[in_Manhattan == 1])
summary(housing_cost[in_Bronx == 1])
summary(housing_cost[in_Brooklyn == 1])
summary(housing_cost[in_Queens == 1])
summary(housing_cost[in_StatenI == 1])
```


Now we create a data frame with our 3 variables, that will be cleaned for NAs and subset which our borough classifications.
```{r}
data_use_prelim <- data.frame(norm_housing_cost, norm_TRANWORK, norm_TRANTIME)
good_obs_data_use <- complete.cases(data_use_prelim,borough_f)
dat_use <- subset(data_use_prelim,good_obs_data_use)
y_use <- subset(borough_f,good_obs_data_use)
```


We select 80% of the data to train the algorithm, and the other 20% to test. 
```{r}
set.seed(12345)
NN_obs <- sum(good_obs_data_use == 1)
select1 <- (runif(NN_obs) < 0.8)
train_data <- subset(dat_use,select1)
test_data <- subset(dat_use,(!select1))
cl_data <- y_use[select1]
true_data <- y_use[!select1]
```


This is the algorithmic portion. We we run our data, we see accuracy K-nn levels as follows:

Test 1:
[1] 1.0000000 0.4090492
[1] 3.000000 0.415775
[1] 5.0000000 0.4203607
[1] 7.0000000 0.4215836
[1] 9.0000000 0.4206665

Test 2:
[1] 1.0000000 0.4154693
[1] 3.0000000 0.4182207
[1] 5.0000000 0.4203607
[1] 7.0000000 0.4212779
[1] 9.0000000 0.4240293

The range seems to be from about 40-43%, not a bad improvement from the the original data. Our commute variables improved the accuracy from the original INCTOT + housing. We also see an upward trend where accuracy increases with the more neighboors our attributes is classified in, at least until k=9 for test 1. Most likely, other variables such as certain races and ehtnicities could have been better to classify within boroughs. However, I wanted to try this one and am not dissapointed at the results. A step further would maybe have been to classify using private and public transport as attritube since we learned they vary thoughout. Adding more variables in general would have helped as well, although that may affect the quality of our test data.

```{r}
summary(cl_data)
prop.table(summary(cl_data))
summary(train_data)
require(class)
for (indx in seq(1, 9, by= 2)) {
 pred_borough <- knn(train_data, test_data, cl_data, k = indx, l = 0, prob = FALSE, use.all = TRUE)
num_correct_labels <- sum(pred_borough == true_data)
correct_rate <- num_correct_labels/length(true_data)
print(c(indx,correct_rate))
}
```




One way we can attempt to classify better is by focusing on neighborhood rather than borough. In terms of commute and housing cost, neighborhood tend to be more homogenous. I wanted to keep looking at Queens since they not only have less public commuters and long commute times (relative to all boroughs except SI), but it's the only borough that at the same time had relatively high housing costs. This makes me think that neighborhoods in Queens are more unequal. This shoud allow us to classify people better.

I chose to compare Astoria & LIC and Flushing to show how different two neighborhoods can be within one borough, affecting the accuracy of our borough test. Regarding our attributes, Astoria & LIC have shorter commute times to Manhattan, higher valued housing, and due to housing density most likely less cars than any other part of Queens. Flushing on the other hands is more in a midpoint in Northern Queens in terms of proximity to Manhattan, and has a lower income level than Astoria as we can see below.

```{r}
summary(INCTOT[dat_NYC$PUMA == 4101])
summary(INCTOT[dat_NYC$PUMA == 4103])
```

We build a new k-nn algorithm to classify New Yorkers within these two neighborhoods.

```{r}
dat_NBD <- subset(dat_NYC, (dat_NYC$PUMA == 4101) | (dat_NYC$PUMA == 4103) | (dat_NYC$PUMA == 4107) & (dat_NYC$AGE > 20) & (dat_NYC$AGE < 66))
                  
attach(dat_NBD)
nbd_f <- factor((PUMA == 4101) + 2*(PUMA == 4103), levels=c(1,2),labels = c("Astoria & LIC","Flushing"))
```

```{r}
OWNCOST <- dat_NBD$OWNCOST
RENT <- dat_NBD$RENT
is.na(OWNCOST) <- which(OWNCOST == 9999999)
housing_cost <- OWNCOST + RENT
norm_housing_cost <- norm_varb(housing_cost)
TRANWORK <- dat_NBD$TRANWORK
TRANWORK <- as.numeric(TRANWORK)
is.na(TRANWORK) <- which(TRANWORK == 0)
norm_TRANWORK <- norm_varb(TRANWORK)
TRANTIME <- dat_NBD$TRANTIME
is.na(TRANTIME) <- which(TRANTIME == 0)
norm_TRANTIME <- norm_varb(TRANTIME)
```


```{r}
data_use_prelim <- data.frame(norm_housing_cost, norm_TRANWORK, norm_TRANTIME)
good_obs_data_use <- complete.cases(data_use_prelim, neighborhood_f)
dat_use <- subset(data_use_prelim,good_obs_data_use)
y_use <- subset(nbd_f,good_obs_data_use)
```

```{r}
set.seed(12345)
NN_obs <- sum(good_obs_data_use == 1)
select1 <- (runif(NN_obs) < 0.8)
train_data <- subset(dat_use,select1)
test_data <- subset(dat_use,(!select1))
cl_data <- y_use[select1]
true_data <- y_use[!select1]
```


As we can see, the accuracy improved dramatically, ranging from 72-76%. 
```{r}
summary(cl_data)
prop.table(summary(cl_data))
summary(train_data)
require(class)
for (indx in seq(1, 9, by= 2)) {
 pred_borough <- knn(train_data, test_data, cl_data, k = indx, l = 0, prob = FALSE, use.all = TRUE)
num_correct_labels <- sum(pred_borough == true_data)
correct_rate <- num_correct_labels/length(true_data)
print(c(indx,correct_rate))
}
```

