# Databricks notebook source
#dbutils.fs.cp("dbfs:/FileStore/tables/load_dataikuapi.py", "dbfs:/Chronos/general/load_dataikuapi.py")

# COMMAND ----------

dbutils.fs.ls("Chronos/data/m5")

# COMMAND ----------

dbutils.fs.ls("FileStore/bbac_forecast")

# COMMAND ----------

#install.packages("fst")

# COMMAND ----------

library(data.table)
library(stringr)
library(fst)
library(readxl)
library(data.table)
library(lubridate)
library(magrittr)
library(feather)
primarykey <- c("snr","es1","es2")

# COMMAND ----------

CJ.dt <- 
  function (X, Y) 
  {
    k <- NULL
    X <- X[, c(k = 1, .SD)]
    setkey(X, k)
    Y <- Y[, c(k = 1, .SD)]
    setkey(Y, NULL)
    X[Y, allow.cartesian = TRUE][, `:=`(k, NULL)]
  }

list_files2 <- function(target_folder,date_flag=T,date_first=F) {
      
      paths <- data.table(full_file=list.files(paste0("/dbfs/Chronos/bbac_forecast/",target_folder),full.names = T),folder=target_folder,filename=list.files(paste0("/dbfs/Chronos/bbac_forecast/",target_folder),full.names = F))
      
      if (date_flag) paths[, date := as.Date(str_sub(str_replace_all(filename,"[^[:digit:]]",""),-8+9*(date_first==T),-1+9*(date_first==T)), format = "%Y%m%d")]
      
      return(paths)
      
   }

# COMMAND ----------

# file_dt = data.table(filepath= list.files("/dbfs/FileStore/tables/",full.names=T),filename=list.files("/dbfs/FileStore/tables/",full.names=F))
# file_dt = file_dt[filename %like% "calloff"]
# file_dt[,target:=paste0("/dbfs/Chronos/bbac_forecast/forecast/invlab_weekly/data/",filename)]
# file_dt[,file.copy(filepath,target)]

# COMMAND ----------

forecast <- list_files2("forecast/invlab_weekly/data") 

forecast_list <- vector(mode="list",length=nrow(forecast))

for (i in 1:nrow(forecast)){
   forecast_list[[i]] <- read.fst(forecast[i,full_file],as.data.table = T)
   forecast_list[[i]][,filename:=forecast[i,filename]]
   forecast_list[[i]][,filedate:=forecast[i,date]]
   }

forecast_dt <- rbindlist(forecast_list,fill=T)
forecast_dt[,docnum:=NULL]

setorderv(forecast_dt,c(primarykey,"CustomerPurchaseOrderDate","CustomerPlant","ItemNumberOfTheUnderlyingCustomerPurchaseOrder"))

forecast_dt <- forecast_dt[substr(snr,1,1) %in%c("A","N","R")]

forecast_dt <- forecast_dt[substr(snr,1,1) %in%c("A","N","R") & !str_detect(substring(snr,2),"[:alpha:]")]

# COMMAND ----------

invlab_select <- forecast_dt[CustomerPlant=="CNB1",tail(.SD,1),by=eval(c(primarykey,"CustomerPlant","ItemNumberOfTheUnderlyingCustomerPurchaseOrder","filedate","binned_weeks"))]
rm(forecast_dt)
#invlab_select <- invlab_select[CustomerPurchaseOrderDate>=(Sys.Date()-months(1))]
#to do: Sales Unit
invlab_select[,uniqueid:=.GRP,by=.(snr,es1,es2,CustomerPlant,filedate)]

invlab_select[,difftime:=difftime(CustomerPurchaseOrderDate,filedate,units="days") %>%  as.numeric]
options(width = 200)
invlab_select <- invlab_select[difftime>-11]
invlab_select[binned_weeks < as.Date("2018-07-01"),binned_weeks:=floor_date(as.Date("2018-07-01"),"weeks")]

# COMMAND ----------

#Ignore Sales Unit
lab_collapse_main <- invlab_select[,lapply(.SD,sum),by=.(snr,es1,es2,CustomerPlant,filedate, uniqueid)
                              ,.SDcols=c("CurrentCumulativeQuantityOrdered","CurrentCumulativeQuantityReceived")]

lab_collapse_single <- invlab_select[,lapply(.SD,sum),by=.(filedate,binned_weeks, uniqueid)
                                   ,.SDcols=c("ReleasedQuantity")]

# COMMAND ----------

lab_correction_one <- invlab_select[filedate < "2020-10-10",.(CustomerPurchaseOrderDate=max(CustomerPurchaseOrderDate)),by=.(snr,es1,es2)]
lab_correction_two <- invlab_select[filedate > "2020-11-19",.(CustomerPurchaseOrderDate=min(CustomerPurchaseOrderDate)),by=.(snr,es1,es2)]

# COMMAND ----------

lab_correction <- merge(lab_correction_one,lab_correction_two,by=primarykey,all=T,suffix=c("min","max"))
lab_correction[,delta:=difftime(CustomerPurchaseOrderDatemax,CustomerPurchaseOrderDatemin,units="days") %>% as.numeric]

# COMMAND ----------

lab_correction_scope <- lab_correction[delta < 60,.(snr,es1,es2)]

# COMMAND ----------

lab_correction_scope_unique <- invlab_select[lab_correction_scope,on=primarykey][between(filedate,"2020-10-10","2020-11-20"),.(uniqueid)]  %>% unique

# COMMAND ----------

lab_all <- lab_collapse_single[,.(uniqueid,filedate)] %>% unique

# COMMAND ----------

query_dates <- seq.Date(from=invlab_select[,min(binned_weeks)],invlab_select[,max(binned_weeks)]+weeks(1),by="weeks") %>% 
   floor_date("weeks")

week_lab_dt <- data.table(date_lab=query_dates)

lab_span <- CJ.dt(lab_all,week_lab_dt)

lab_span <- lab_collapse_single[,.(uniqueid,date_lab=binned_weeks,ReleasedQuantity)][lab_span,on=c("uniqueid","date_lab")]
# lab_span <- lab_collapse_single[,.(snr, es1, es2, date_lab=binned_weeks,ReleasedQuantity)][lab_span,on=c("snr","es1","es2","date_lab")]


# COMMAND ----------

for (j in names(lab_span)[names(lab_span) %like% "Released"]) set(lab_span,which(is.na(lab_span[[j]])),j,0)

#lab_span[,file_min:=filedate-60]
#lab_span <- lab_span[date_lab < file_min ]
#lab_span[,file_min:=NULL]
#lab_span %>% write.fst("C:/temp/dataiku_mirror/adhoc_analysis/20210324_cashflow/lab.fst")
# rm(invlab_select)


# COMMAND ----------

lab_span[lab_correction_scope_unique,on="uniqueid",ReleasedQuantity:=NA]

# COMMAND ----------

rm(lab_collpase_single)
rm(lab_all)

# COMMAND ----------

lab_raw <- lab_collapse_main[,.(snr,es1,es2,uniqueid)][lab_span,on=.NATURAL]
#rm(lab_span)

# COMMAND ----------

#lab_raw[,count:=.N,by=.(snr,es1,es2,date_lab,filedate)]
#lab_raw[count>1]
lab_raw[,count:=NULL]

# COMMAND ----------

lab_raw[,deliver2fcst:=difftime(date_lab,filedate,units="weeks") %>%  as.numeric]
lab_raw[,bracket:=cut(deliver2fcst/4,c(-Inf,-1,0,1,2,3, Inf))]
lab_raw[,bracket:=str_replace_all(bracket,"\\(|\\]","")]
lab_raw[,bracket:=str_replace_all(bracket,",","_")]
lab_raw[,bracket:=str_replace_all(bracket," ","")]

# lab_raw[snr=="A0000005200"][ReleasedQuantity > 0]

# lab_all <- lab_collapse_single[,.(uniqueid,filedate,CustomerPurchaseOrderDate)] %>% unique

# invlab_select[,range(binned_weeks)]

# query_dates <- seq.Date(from=invlab_select[,min(binned_weeks)],invlab_select[,max(binned_weeks)]+weeks(1),by="weeks") %>% 
#    floor_date("weeks")

# COMMAND ----------

lab_raw[, snr_new := paste0(snr, "_", es1, "_", es2)]
lab_raw[, `:=` (snr = NULL, es1 = NULL, es2 = NULL, deliver2fcst = NULL)]

# COMMAND ----------

rm(lab_collapse_single)
rm(invlab_select)
rm(lab_correction_scope)
rm(lab_correction_two)
rm(lab_correction_one)
rm(lab_correction)

# COMMAND ----------

## fwrite(lab_raw, "/dbfs/Chronos/data/m5/lab_raw.csv", row.names=F)
## Speichern mit arrow funktioniert nicht richtig --> feather ##
## feather::write_feather(lab_raw, "/dbfs/Chronos/data/m5/lab_raw_feather.feather")

# COMMAND ----------

#library(arrow)

# COMMAND ----------

dbutils.fs.mkdirs("Chronos/bbac_forecast/start_fst")
file_dt = data.table(filepath= list.files("/dbfs/FileStore/tables/",full.names=T),filename=list.files("/dbfs/FileStore/tables/",full.names=F))
file_dt = file_dt[(filename %like% "verwendung")]
file_dt = file_dt[,target:=paste0("/dbfs/Chronos/bbac_forecast/start_fst/",filename)]
file_dt[,file.copy(filepath,target)]
file_dt

# COMMAND ----------

start_combo <-list_files2("start_fst")[filename %like% "fst"][.N,full_file] %>% read.fst(as.data.table=T)

# COMMAND ----------

## Invoices ##
# dbutils.fs.rm("Chronos/bbac_forecast/invoice_out/data", TRUE)
dbutils.fs.mkdirs("Chronos/bbac_forecast/invoice_out/data")
dbutils.fs.mkdirs("Chronos/bbac_forecast/selling_prices/data")

# COMMAND ----------



# COMMAND ----------

file_dt = data.table(filepath= list.files("/dbfs/FileStore/tables/",full.names=T),filename=list.files("/dbfs/FileStore/tables/",full.names=F))
file_dt = file_dt[(filename %like% "tp_out")]
file_dt = file_dt[,target:=paste0("/dbfs/Chronos/bbac_forecast/selling_prices/data/",filename)]
file_dt[,file.copy(filepath,target)]
file_dt


# COMMAND ----------

 dbutils.fs.mkdirs("Chronos/bbac_forecast/other")

# COMMAND ----------

file_dt = data.table(filepath= list.files("/dbfs/FileStore/tables/",full.names=T),filename=list.files("/dbfs/FileStore/tables/",full.names=F))
file_dt = file_dt[(filename %like% "mengeadjusted")]
file_dt = file_dt[,target:=paste0("/dbfs/Chronos/bbac_forecast/other/",filename)]
file_dt[,file.copy(filepath,target)]
file_dt


# COMMAND ----------

invmenge <- list_files2("other")[full_file %like% "-1"][.N,full_file] %>% read.fst(as.data.table=T)
invmenge_sub <- invmenge[,.(snr,es1,es2,menge_p)] %>% unique

# COMMAND ----------

#invmenge_sub[,count:=.N,by=.(snr,es1,es2)]
#invmenge_sub[count>1]

# COMMAND ----------

# file_dt = data.table(filepath= list.files("/dbfs/FileStore/tables/",full.names=T),filename=list.files("/dbfs/FileStore/tables/",full.names=F))
# file_dt = file_dt[(filename %like% "inv") & !(filename %like% "old")]
# file_dt = file_dt[,target:=paste0("/dbfs/Chronos/bbac_forecast/invoice_out/data/",filename)]
# file_dt[,file.copy(filepath,target)]
## Falls neue Daten hinzugefügt wurden müssen die alten und neuen Dateien umbennant werden (mit dbutils.fs.mv) ##
## z.B.: inv_20211231.fst -->  inv_20211231_old.fst
##       inv_20211231-1.fst --> inv_20211231.fst

# COMMAND ----------

selling_price <- list_files2("selling_prices/data")[.N,full_file] %>% read.fst(as.data.table=T)
selling_price

# COMMAND ----------

inv_paths <- list_files2("invoice_out/data")[dplyr::between(date,as.Date("2015-12-31"),Sys.Date()+years(1))]
inv_paths

# COMMAND ----------

inv_list  <- vector(mode="list",length=inv_paths %>%  nrow)

for (i in 1:nrow(inv_paths)){
   inv_list[[i]] <- read.fst(inv_paths[i,full_file],as.data.table = T)
}
inv <- rbindlist(inv_list)

# COMMAND ----------

#Offenes Thema: Gutschriften
#ratios <- inv[,.(revenue=sum(menge*betrag)),keyby=.(year(fakturadatum),menge>0)][,abs(min(revenue))/(max(revenue)+abs(min(revenue))),by=year]
#ratio_adjust <- ratio[year==2020,V1]

# COMMAND ----------

inv <- inv[menge>0] 

# COMMAND ----------

inv[,parts_by_parts_scope:=F]
inv[fkart %in% c("ZAFL","ZBS3","ZAGR")  & substr(snr,1,1) %in% c("A","N","R") ,parts_by_parts_scope:=T]
#inv[,.N,by=.(type,fkart,parts_by_parts_scope)]
inv <- inv[auftrgeb %in% c("81900311") & parts_by_parts_scope ==T]
inv[,fakturadatum_rounded:=floor_date(fakturadatum,"weeks")]
inv[,max_fak:=max(fakturadatum_rounded),by=.(snr,es1,es2)]

# COMMAND ----------

inv_sub <- inv[,.(menge=sum(menge),betrag=weighted.mean(betrag,abs(menge))),keyby=.(snr,es1,es2,fakturadatum_rounded)]
inv_sub[,range(fakturadatum_rounded)]
#rm(inv)
#inv_sub[,.(min(inv_sub$fakturadatum_rounded),max(inv_sub$fakturadatum_rounded))] %>%  unique
#inv_month <- inv_sub[,.(y=sum(menge*betrag)),keyby=.(ds=floor_date(fakturadatum_rounded,"months"))]

query_dates <- seq.Date(from=inv_sub[,min(fakturadatum_rounded)],inv_sub[,max(fakturadatum_rounded)]+weeks(1),by="weeks") %>% 
   floor_date("weeks")

# COMMAND ----------

inv_sub[,betrag:=data.table::nafill(betrag,type="locf"),by=.(snr,es1,es2)]

# COMMAND ----------

inv_sub[,c("vp_start","vp_end"):=fakturadatum_rounded]

# COMMAND ----------

setkeyv(inv_sub,c(primarykey,"vp_start","vp_end"))

# COMMAND ----------

selling_price_short <- selling_price[,.(snr,es1,es2,vp_start,vp_end,price_vp)]

# COMMAND ----------

setkeyv(selling_price_short,c(primarykey,"vp_start","vp_end"))

# COMMAND ----------

inv_sub2 <- foverlaps(inv_sub,selling_price_short)

# COMMAND ----------

inv_sub2_nomiss <- inv_sub2[!is.na(price_vp)]
inv_sub2_miss <- inv_sub2[is.na(price_vp),names(inv_sub),with=F]

# COMMAND ----------

inv_sub2_miss <- selling_price_short[inv_sub2_miss,on=c("snr","es1","es2","vp_start"),roll="nearest"]

# COMMAND ----------

inv_sub2 <- rbind(inv_sub2_nomiss[,!names(inv_sub2_nomiss) %like% "i.|^vp_",with=F],inv_sub2_miss[,!names(inv_sub2_miss) %like% "i.|^vp_",with=F])

# COMMAND ----------

inv_sub2[,delta:=betrag/shift(betrag),by=.(snr,es1,es2)]

# COMMAND ----------

master_interval <- rbind(inv_sub2[,.(snr,es1,es2)] %>%  unique)
week_dt <- data.table(fakturadatum_rounded=query_dates)

# COMMAND ----------

fcst_span <- CJ.dt(master_interval,week_dt) %>% unique


# COMMAND ----------

inv_sub2 <- invmenge_sub[inv_sub,on=primarykey]

# COMMAND ----------

inv_sub2[is.na(menge_p),menge_p:=1]

# COMMAND ----------

inv_sub2[,menge_adjusted:=menge/menge_p]

# COMMAND ----------

inv_sub2[,.(sd(menge_adjusted)/mean(menge),sd(menge)/mean(menge))]

# COMMAND ----------

##Megren führt zu NAs bei Betrag --> wird in den nächsten Schritten mit 0 gefüllt ##
fcst_span2 <- merge(fcst_span,inv_sub2,by=c(primarykey,"fakturadatum_rounded"),all.x=T)
#rm(fcst_span)
#rm(inv_sub)
#rm(master_interval)
#gc()

# COMMAND ----------

fcst_span2[,betrag:=data.table::nafill(betrag,type="locf"),by=.(snr,es1,es2)]
# fcst_span2[,betrag:=data.table::nafill(betrag,type="nocb"),by=.(snr,es1,es2)]

# COMMAND ----------

#fcst_span2[,lifecycle_duration:=difftime(max(fakturadatum_rounded),min(fakturadatum_rounded),units="days"),by=.(snr_new)]


# COMMAND ----------

#fcst_span2[sample(1:.N,1),.(snr_new)] %>% unique %>% fcst_span2[.,on=.NATURAL]

# COMMAND ----------

#fcst_span2[dplyr::near(betrag,0.06),.(snr_new)]  %>% unique  %>% fcst_span2[.,on="snr_new"] %>% .[dplyr::near(betrag,0.07),.(snr_new)]  %>% unique %>% fcst_span2[.,on="snr_new"]  %>% .[dplyr::near(betrag,20.00),.(snr_new)]  %>% unique %>% fcst_span2[.,on="snr_new"]

# COMMAND ----------

#fcst_span2[snr_new=="N914007006046_NA_NA"][menge>0]

# COMMAND ----------

# feather::write_feather(fcst_span2, "/dbfs/Chronos/data/m5/inv_unique_einseitig_aufgerollt.feather")

# COMMAND ----------



# COMMAND ----------

for (j in names(fcst_span2)[names(fcst_span2) %like% "menge|betrag"]) set(fcst_span2,which(is.na(fcst_span2[[j]])),j,0)

fcst_span2[, snr_new := as.character(paste0(snr, "_", es1, "_", es2))]
fcst_span2[,`:=` (snr = NULL, es1 = NULL, es2 = NULL)]
str(fcst_span2)

# COMMAND ----------

fcst_span2_agg = fcst_span2[, sum(menge*betrag), keyby = .(floor_date(fakturadatum_rounded,"months"))]
fcst_span2_agg %>% tail(20)

# COMMAND ----------

fcst_span2_agg


# COMMAND ----------



# COMMAND ----------

feather::write_feather(fcst_span2, "/dbfs/Chronos/data/m5/inv_unique.feather")
fcst_span3 <- fcst_span2[,.(snr_new,betrag,menge=menge_adjusted,fakturadatum_rounded)]
feather::write_feather(fcst_span3, "/dbfs/Chronos/data/m5/inv_unique_adjusted.feather")

# COMMAND ----------

# # # Create Subsample Invoices Only ##
# inv_snr = fcst_span2[, snr_new] %>% unique
# inv_snr_reduced = inv_snr[1:floor(0.1*length(inv_snr))]
# inv_reduced2 = fcst_span2[snr_new %in% inv_snr_reduced]
# feather::write_feather(inv_reduced2, "/dbfs/Chronos/data/m5/inv_reduced2_einseitig_aufgerollt.feather")

# COMMAND ----------

#lab_snr_unique = lab_raw$snr_new %>% unique
#help = fcst_span2[!(snr_new %in% lab_snr_unique)]

# COMMAND ----------

# ## Create Subsample for peformance reasons ##
# snr_combined <- intersect(unique(lab_raw$snr_new), unique(fcst_span2$snr_new))
# snr_combined_reduced <- snr_combined[1:floor(0.1*length(snr_combined))]
# #fwrite(data.table(snr_combined_reduced), "/dbfs/Chronos/data/m5/snr_subset.csv")

# help <- fcst_span2[, .(snr_new, fakturadatum_rounded)] %>% unique
# snr_combined_reduced <- help[snr_new %in% snr_combined_reduced,.N, by = "snr_new"] %>% .[N == 191, snr_new] 
# rm(help)
# rm(snr_combined)
# gc()
# lab_reduced = lab_raw[snr_new %in% snr_combined_reduced]
# inv_reduced = fcst_span2[snr_new %in% snr_combined_reduced]

# # feather::write_feather(lab_reduced, "/dbfs/Chronos/data/m5/lab_reduced.feather")
# # feather::write_feather(inv_reduced, "/dbfs/Chronos/data/m5/inv_reduced.feather")