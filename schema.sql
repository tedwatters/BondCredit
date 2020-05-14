-- MySQL dump 10.13  Distrib 5.7.30, for Linux (x86_64)
--
-- Host: localhost    Database: bond_credit
-- ------------------------------------------------------
-- Server version	5.7.30-0ubuntu0.18.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `issue`
--

DROP TABLE IF EXISTS `issue`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `issue` (
  `ISSUE_ID` int(11) DEFAULT NULL,
  `ISSUER_ID` int(11) DEFAULT NULL,
  `PROSPECTUS_ISSUER_NAME` text,
  `ISSUER_CUSIP` text,
  `ISSUE_CUSIP` text,
  `ISSUE_NAME` text,
  `MATURITY` int(11) DEFAULT NULL,
  `DEFAULTED` text,
  `TENDER_EXCH_OFFER` text,
  `ANNOUNCED_CALL` text,
  `ACTIVE_ISSUE` text,
  `BOND_TYPE` text,
  `ISIN` text,
  `COMPLETE_CUSIP` text,
  `ACTION_TYPE` text,
  `EFFECTIVE_DATE` int(11) DEFAULT NULL,
  `CROSS_DEFAULT` text,
  `RATING_DECLINE_TRIGGER_PUT` text,
  `RATING_DECLINE_PROVISION` text,
  `FUNDED_DEBT_IS` text,
  `INDEBTEDNESS_IS` text,
  `FUNDED_DEBT_SUB` text,
  `INDEBTEDNESS_SUB` text,
  `CUSIP_NAME` text,
  `INDUSTRY_GROUP` int(11) DEFAULT NULL,
  `INDUSTRY_CODE` int(11) DEFAULT NULL,
  `ESOP` text,
  `IN_BANKRUPTCY` text,
  `PARENT_ID` text,
  `NAICS_CODE` text,
  `COUNTRY_DOMICILE` text,
  `SIC_CODE` text,
  KEY `idx_issue_COMPLETE_CUSIP` (`COMPLETE_CUSIP`(9))
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `main`
--

DROP TABLE IF EXISTS `main`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `main` (
  `DEFAULTED` text,
  `TENDER_EXCH_OFFER` text,
  `ANNOUNCED_CALL` text,
  `ACTIVE_ISSUE` text,
  `BOND_TYPE` text,
  `ISIN` text,
  `ACTION_TYPE` text,
  `EFFECTIVE_DATE` int(11) DEFAULT NULL,
  `CROSS_DEFAULT` text,
  `RATING_DECLINE_TRIGGER_PUT` text,
  `RATING_DECLINE_PROVISION` text,
  `FUNDED_DEBT_IS` text,
  `INDEBTEDNESS_IS` text,
  `FUNDED_DEBT_SUB` text,
  `INDEBTEDNESS_SUB` text,
  `CUSIP_NAME` text,
  `INDUSTRY_GROUP` int(11) DEFAULT NULL,
  `INDUSTRY_CODE` int(11) DEFAULT NULL,
  `ESOP` text,
  `IN_BANKRUPTCY` text,
  `PARENT_ID` text,
  `NAICS_CODE` text,
  `COUNTRY_DOMICILE` text,
  `SIC_CODE` text,
  `ISSUE_ID` int(11) DEFAULT NULL,
  `RATING_TYPE` text,
  `RATING_DATE` int(11) DEFAULT NULL,
  `RATING` text,
  `RATING_STATUS` text,
  `REASON` text,
  `RATING_STATUS_DATE` text,
  `INVESTMENT_GRADE` text,
  `ISSUER_ID` int(11) DEFAULT NULL,
  `PROSPECTUS_ISSUER_NAME` text,
  `ISSUER_CUSIP` text,
  `ISSUE_CUSIP` text,
  `ISSUE_NAME` text,
  `MATURITY` int(11) DEFAULT NULL,
  `OFFERING_DATE` int(11) DEFAULT NULL,
  `COMPLETE_CUSIP` text,
  KEY `idx_main_COMPLETE_CUSIP` (`COMPLETE_CUSIP`(9)),
  KEY `idx_main_RATING_DATE` (`RATING_DATE`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `main_n_index`
--

DROP TABLE IF EXISTS `main_n_index`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `main_n_index` (
  `DEFAULTED` text,
  `TENDER_EXCH_OFFER` text,
  `ANNOUNCED_CALL` text,
  `ACTIVE_ISSUE` text,
  `BOND_TYPE` text,
  `ISIN` text,
  `ACTION_TYPE` text,
  `EFFECTIVE_DATE` int(11) DEFAULT NULL,
  `CROSS_DEFAULT` text,
  `RATING_DECLINE_TRIGGER_PUT` text,
  `RATING_DECLINE_PROVISION` text,
  `FUNDED_DEBT_IS` text,
  `INDEBTEDNESS_IS` text,
  `FUNDED_DEBT_SUB` text,
  `INDEBTEDNESS_SUB` text,
  `CUSIP_NAME` text,
  `INDUSTRY_GROUP` int(11) DEFAULT NULL,
  `INDUSTRY_CODE` int(11) DEFAULT NULL,
  `ESOP` text,
  `IN_BANKRUPTCY` text,
  `PARENT_ID` text,
  `NAICS_CODE` text,
  `COUNTRY_DOMICILE` text,
  `SIC_CODE` text,
  `ISSUE_ID` int(11) DEFAULT NULL,
  `RATING_TYPE` text,
  `RATING_DATE` int(11) DEFAULT NULL,
  `RATING` text,
  `RATING_STATUS` text,
  `REASON` text,
  `RATING_STATUS_DATE` text,
  `INVESTMENT_GRADE` text,
  `ISSUER_ID` int(11) DEFAULT NULL,
  `PROSPECTUS_ISSUER_NAME` text,
  `ISSUER_CUSIP` text,
  `ISSUE_CUSIP` text,
  `ISSUE_NAME` text,
  `MATURITY` int(11) DEFAULT NULL,
  `OFFERING_DATE` int(11) DEFAULT NULL,
  `COMPLETE_CUSIP` text,
  `N_INDEX` bigint(22) NOT NULL DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `ratings`
--

DROP TABLE IF EXISTS `ratings`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ratings` (
  `ISSUE_ID` int(11) DEFAULT NULL,
  `RATING_TYPE` text,
  `RATING_DATE` int(11) DEFAULT NULL,
  `RATING` text,
  `RATING_STATUS` text,
  `REASON` text,
  `RATING_STATUS_DATE` text,
  `INVESTMENT_GRADE` text,
  `ISSUER_ID` int(11) DEFAULT NULL,
  `PROSPECTUS_ISSUER_NAME` text,
  `ISSUER_CUSIP` text,
  `ISSUE_CUSIP` text,
  `ISSUE_NAME` text,
  `MATURITY` int(11) DEFAULT NULL,
  `OFFERING_DATE` int(11) DEFAULT NULL,
  `COMPLETE_CUSIP` text,
  KEY `idx_ratings_COMPLETE_CUSIP` (`COMPLETE_CUSIP`(9))
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-14 14:40:46
