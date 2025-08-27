#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct CAPE Converter - English Version
Directly converts CAPE JSON to English Markdown reports
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
import re

class DirectCapeConverterEnglish:
    """Direct converter from CAPE JSON to English Markdown reports"""
    
    def __init__(self):
        # Load 21 critical API data
        self.critical_apis_data = self._load_critical_apis_data()
        self._build_api_operation_mappings()
        self._build_sensitive_patterns()
        
        # Build API description dictionary from 21 API data
        self.api_descriptions = {}
        for api_name, api_info in self.critical_apis_data.items():
            self.api_descriptions[api_name] = api_info.get('description', 'API function')
    
    def _build_api_operation_mappings(self):
        """Builds mappings from specific APIs to generic operation types."""
        self.api_op_mappings = {
            'registry': {
                'write': {'RegSetValueExW', 'RegCreateKeyExW', 'NtSetValueKey'},
                'read': {'RegQueryValueExW', 'RegOpenKeyExW', 'NtQueryValueKey', 'NtOpenKey'},
                'delete': {'RegDeleteKeyW', 'RegDeleteValueW', 'NtDeleteKey'}
            },
            'file': {
                'write': {'NtCreateFile', 'NtWriteFile', 'WriteFile', 'CopyFileW', 'MoveFileW'},
                'read': {'NtOpenFile', 'NtReadFile', 'ReadFile'},
                'delete': {'NtDeleteFile', 'DeleteFileW'}
            }
        }

    def _build_sensitive_patterns(self):
        """Build knowledge base of sensitive file and registry patterns, categorized by operation type."""
        self.sensitive_patterns = {
            'registry': {
                'write': {
                    re.compile(r'\\Software\\Microsoft\\Windows\\CurrentVersion\\Run(Once)?(\\[^\\]+)?$', re.I): {
                        'description': 'Addition of a new program to the Run/RunOnce keys for automatic startup.',
                        'category_tag': 'Persistence'
                    },
                    re.compile(r'\\Software\\Microsoft\\Windows\\CurrentVersion\\RunServices(Once)?(\\[^\\]+)?$', re.I): {
                        'description': 'Creation of a RunServices-based startup entry.',
                        'category_tag': 'Persistence'
                    },
                    re.compile(r'\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Winlogon\\(Shell|Userinit)$', re.I): {
                        'description': 'Modification of Winlogon Shell/Userinit for persistence or shell replacement.',
                        'category_tag': 'Persistence'
                    },
                    re.compile(r'\\System\\CurrentControlSet\\Services\\', re.I): {
                        'description': 'Creation or modification of a system service.',
                        'category_tag': 'Persistence'
                    },
                    re.compile(r'\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Image File Execution Options\\', re.I): {
                        'description': 'Use of Image File Execution Options for process hijacking or debugging.',
                        'category_tag': 'Defense Evasion'
                    },
                    re.compile(r'\\Software\\Microsoft\\Windows\\CurrentVersion\\Policies\\(System|Explorer)', re.I): {
                        'description': 'Modification of system/explorer policies (e.g., disable tools or UAC tweaks).',
                        'category_tag': 'Defense Evasion'
                    },
                    re.compile(r'\\System\\CurrentControlSet\\Control\\Lsa\\(Security Packages|Notification Packages)$', re.I): {
                        'description': 'Modification of LSA packages to load a malicious DLL.',
                        'category_tag': 'Credential Access'
                    },
                    re.compile(r'\\Software\\Classes\\CLSID\\\{[0-9A-Fa-f\-]+\}\\InprocServer32$', re.I): {
                        'description': 'COM Hijacking via InprocServer32 value.',
                        'category_tag': 'Persistence'
                    },
                    re.compile(r'\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\User Shell Folders\\Startup$', re.I): {
                        'description': 'Modification of Startup folder shell mapping.',
                        'category_tag': 'Persistence'
                    },
                    re.compile(r'\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Advanced$', re.I): {
                        'description': 'Tuning Explorer advanced settings possibly to hide files or extensions.',
                        'category_tag': 'Defense Evasion'
                    },
                },
                'read': {
                    re.compile(r'\\System\\CurrentControlSet\\Services\\', re.I): {
                        'description': 'Enumeration of system services.',
                        'category_tag': 'Discovery'
                    },
                    re.compile(r'\\Software\\Microsoft\\Windows\\CurrentVersion\\Run(Once)?(\\[^\\]+)?$', re.I): {
                        'description': 'Querying startup entries from Run/RunOnce.',
                        'category_tag': 'System Query'
                    },
                    re.compile(r'\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Winlogon\\(Shell|Userinit)$', re.I): {
                        'description': 'Querying Winlogon Shell/Userinit settings.',
                        'category_tag': 'System Query'
                    },
                    re.compile(r'\\Software\\Microsoft\\Windows\\CurrentVersion\\Policies\\', re.I): {
                        'description': 'Querying system policy keys.',
                        'category_tag': 'System Query'
                    },
                    re.compile(r'\\System\\CurrentControlSet\\Control\\Lsa\\', re.I): {
                        'description': 'Querying LSA configuration.',
                        'category_tag': 'System Query'
                    },
                    re.compile(r'\\Software\\Microsoft\\Windows NT\\CurrentVersion\\Image File Execution Options\\', re.I): {
                        'description': 'Querying IFEO entries.',
                        'category_tag': 'Discovery'
                    },
                },
                'delete': {
                    re.compile(r'\\Software\\Microsoft\\Windows\\CurrentVersion\\Run(Once)?\\[^\\]+$', re.I): {
                        'description': 'Deletion of a startup entry.',
                        'category_tag': 'Defense Evasion'
                    },
                    re.compile(r'\\System\\CurrentControlSet\\Services\\[^\\]+$', re.I): {
                        'description': 'Deletion of a service registration.',
                        'category_tag': 'Defense Evasion'
                    },
                }
            },
            'file': {
                'write': {
                    re.compile(r'\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Startup\\', re.I): {
                        'description': 'Creation of a file in the Startup folder.',
                        'category_tag': 'Persistence'
                    },
                    re.compile(r'C:\\Windows\\System32\\drivers\\etc\\hosts$', re.I): {
                        'description': 'Modification of the hosts file to hijack network traffic.',
                        'category_tag': 'Defense Evasion'
                    },
                    re.compile(r'\.lnk$', re.I): {
                        'description': 'Creation of a shortcut file (.lnk).',
                        'category_tag': 'Execution'
                    },
                    re.compile(r'(C:\\Users\\|C:\\Documents and Settings\\)[^\\]+\\', re.I): {
                        'description': 'Creation of a file in a user profile directory.',
                        'category_tag': 'Execution'
                    },
                    re.compile(r'C:\\Windows\\(Temp|Prefetch|System32|SysWOW64)\\', re.I): {
                        'description': 'Creation of a file in a sensitive system directory.',
                        'category_tag': 'Execution'
                    },
                    re.compile(r'C:\\Windows\\System32\\Tasks\\', re.I): {
                        'description': 'Creation or modification of a Scheduled Task.',
                        'category_tag': 'Persistence'
                    }
                },
                'read': {
                    re.compile(r'\\SAM|\\SECURITY|\\SYSTEM$', re.I): {
                        'description': 'Access to sensitive registry hives stored as files.',
                        'category_tag': 'Credential Access'
                    },
                    re.compile(r'C:\\ProgramData\\', re.I): {
                        'description': 'Reading shared application data.',
                        'category_tag': 'System Query'
                    },
                    re.compile(r'\\AppData\\(Roaming|Local)\\', re.I): {
                        'description': 'Reading user application data.',
                        'category_tag': 'Application Runtime'
                    },
                    re.compile(r'C:\\Windows\\(Prefetch|WinSxS)\\', re.I): {
                        'description': 'Reading system component caches or assemblies.',
                        'category_tag': 'Discovery'
                    }
                },
                'delete': {
                    re.compile(r'\.log$', re.I): {
                        'description': 'Deletion of log files.',
                        'category_tag': 'Defense Evasion'
                    },
                    re.compile(r'C:\\Windows\\System32\\Tasks\\', re.I): {
                        'description': 'Deletion of a Scheduled Task file.',
                        'category_tag': 'Defense Evasion'
                    }
                }
            }
        }
        self.benign_patterns = {
            'registry': {
                'write': {
                    re.compile(r'\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\', re.I): {
                        'description': 'Creation of an Uninstall entry.',
                        'category_tag': 'Installer'
                    },
                },
                'read': {
                    re.compile(r'\\Software\\Classes\\', re.I): {
                        'description': 'Querying file associations or COM objects.',
                        'category_tag': 'System Query'
                    },
                    re.compile(r'\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\', re.I): {
                        'description': 'Querying user interface settings (e.g., shell folders).',
                        'category_tag': 'System Query'
                    },
                    re.compile(r'\\System\\CurrentControlSet\\Control\\Nls\\Language\\', re.I): {
                        'description': 'Querying system language and locale settings.',
                        'category_tag': 'System Query'
                    },
                    re.compile(r'\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders', re.I): {
                        'description': 'Querying shell folder locations.',
                        'category_tag': 'System Query'
                    },
                    re.compile(r'\\Software\\Microsoft\\Windows\\CurrentVersion\\Run(Once)?(\\[^\\]+)?$', re.I): {
                        'description': 'Querying startup entries for configuration checks.',
                        'category_tag': 'System Query'
                    }
                },
                'delete': {}
            },
            'file': {
                'write': {
                    re.compile(r'C:\\Program Files( \(x86\))?\\', re.I): {
                        'description': 'Creation of files or directories in the Program Files folder.',
                        'category_tag': 'Installer'
                    },
                    re.compile(r'\.log$', re.I): {
                        'description': 'Creation or writing to a log file.',
                        'category_tag': 'Application Runtime'
                    },
                     re.compile(r'\\AppData\\Local\\Temp\\', re.I): {
                        'description': 'Creation of temporary files.',
                        'category_tag': 'Application Runtime'
                    },
                    re.compile(r'\\AppData\\Roaming\\[^\\]+\\', re.I): {
                         'description': 'Writing application data under Roaming profile.',
                         'category_tag': 'Application Runtime'
                    }
                },
                'read': {
                    re.compile(r'C:\\Windows\\(System32|SysWOW64)\\drivers\\', re.I): {
                        'description': 'Accessing system drivers.',
                        'category_tag': 'System Query'
                    },
                    re.compile(r'\\AppData\\Local\\', re.I): {
                        'description': 'Reading application-specific data.',
                        'category_tag': 'Application Runtime'
                    }
                },
                'delete': {}
            }
        }
    
    def _get_critical_apis(self) -> Dict[str, Dict]:
        """Get detailed data for 21 sensitive APIs"""
        return {
            'CreateProcessW': {
                'category': 'Process Control',
                'privilege_level': 'User-level',
                'description': 'Create new process',
                'risk_level': 'High',
                'threat_assessment': 'Can be used to execute malicious programs',
                'attack_technique': 'Process Injection',
                'mitre_technique': 'T1055',
                'malicious_score': 8.5
            },
            'VirtualAllocEx': {
                'category': 'Memory Management',
                'privilege_level': 'User-level',
                'description': 'Allocate memory in other processes',
                'risk_level': 'High',
                'threat_assessment': 'Key step in process injection',
                'attack_technique': 'Process Injection',
                'mitre_technique': 'T1055',
                'malicious_score': 9.0
            },
            'WriteProcessMemory': {
                'category': 'Memory Management',
                'privilege_level': 'User-level',
                'description': 'Write to other process memory',
                'risk_level': 'High',
                'threat_assessment': 'Key step in process injection',
                'attack_technique': 'Process Injection',
                'mitre_technique': 'T1055',
                'malicious_score': 9.2
            },
            'CreateRemoteThread': {
                'category': 'Thread Control',
                'privilege_level': 'User-level',
                'description': 'Create thread in other processes',
                'risk_level': 'High',
                'threat_assessment': 'Key step in process injection',
                'attack_technique': 'Process Injection',
                'mitre_technique': 'T1055',
                'malicious_score': 9.1
            },
            'SetWindowsHookExW': {
                'category': 'System Hook',
                'privilege_level': 'User-level',
                'description': 'Install system hook',
                'risk_level': 'High',
                'threat_assessment': 'Can be used for keylogging or DLL injection',
                'attack_technique': 'DLL Injection',
                'mitre_technique': 'T1055.001',
                'malicious_score': 8.7
            },
            'LoadLibraryW': {
                'category': 'Module Loading',
                'privilege_level': 'User-level',
                'description': 'Dynamically load DLL',
                'risk_level': 'Medium',
                'threat_assessment': 'Can be used to load malicious DLL',
                'attack_technique': 'DLL Injection',
                'mitre_technique': 'T1055.001',
                'malicious_score': 7.5
            },
            'GetProcAddress': {
                'category': 'Module Management',
                'privilege_level': 'User-level',
                'description': 'Get function address',
                'risk_level': 'Medium',
                'threat_assessment': 'Key step in dynamic API calling',
                'attack_technique': 'API Hiding',
                'mitre_technique': 'T1027.007',
                'malicious_score': 7.0
            },
            'RegSetValueExW': {
                'category': 'Registry Operation',
                'privilege_level': 'User-level',
                'description': 'Set registry value',
                'risk_level': 'Medium',
                'threat_assessment': 'Can be used for persistence or configuration modification',
                'attack_technique': 'Registry Persistence',
                'mitre_technique': 'T1547.001',
                'malicious_score': 7.8
            },
            'CreateFileW': {
                'category': 'File Operation',
                'privilege_level': 'User-level',
                'description': 'Create or open file',
                'risk_level': 'Medium',
                'threat_assessment': 'Can be used for file operations or creating malicious files',
                'attack_technique': 'File Operation',
                'mitre_technique': 'T1005',
                'malicious_score': 6.5
            },
            'WriteFile': {
                'category': 'File Operation',
                'privilege_level': 'User-level',
                'description': 'Write file data',
                'risk_level': 'Medium',
                'threat_assessment': 'Can be used to create malicious files or modify system files',
                'attack_technique': 'File Operation',
                'mitre_technique': 'T1005',
                'malicious_score': 6.8
            },
            'CryptEncrypt': {
                'category': 'Encryption Operation',
                'privilege_level': 'User-level',
                'description': 'Encrypt data',
                'risk_level': 'High',
                'threat_assessment': 'Can be used for ransomware or data encryption',
                'attack_technique': 'Data Encryption',
                'mitre_technique': 'T1486',
                'malicious_score': 8.9
            },
            'InternetOpenW': {
                'category': 'Network Communication',
                'privilege_level': 'User-level',
                'description': 'Initialize WinINet',
                'risk_level': 'Medium',
                'threat_assessment': 'Initial step in network communication',
                'attack_technique': 'C2 Communication',
                'mitre_technique': 'T1071.001',
                'malicious_score': 7.2
            },
            'InternetOpenUrlW': {
                'category': 'Network Communication',
                'privilege_level': 'User-level',
                'description': 'Open network URL',
                'risk_level': 'Medium',
                'threat_assessment': 'Can be used to download malicious files or C2 communication',
                'attack_technique': 'C2 Communication',
                'mitre_technique': 'T1071.001',
                'malicious_score': 7.6
            },
            'HttpSendRequestW': {
                'category': 'Network Communication',
                'privilege_level': 'User-level',
                'description': 'Send HTTP request',
                'risk_level': 'Medium',
                'threat_assessment': 'Can be used for data exfiltration or C2 communication',
                'attack_technique': 'C2 Communication',
                'mitre_technique': 'T1071.001',
                'malicious_score': 7.4
            },
            'WSAStartup': {
                'category': 'Network Communication',
                'privilege_level': 'User-level',
                'description': 'Initialize Winsock',
                'risk_level': 'Medium',
                'threat_assessment': 'Basic API for network communication',
                'attack_technique': 'C2 Communication',
                'mitre_technique': 'T1071',
                'malicious_score': 6.8
            },
            'socket': {
                'category': 'Network Communication',
                'privilege_level': 'User-level',
                'description': 'Create socket',
                'risk_level': 'Medium',
                'threat_assessment': 'Basic API for network communication',
                'attack_technique': 'C2 Communication',
                'mitre_technique': 'T1071',
                'malicious_score': 6.9
            },
            'connect': {
                'category': 'Network Communication',
                'privilege_level': 'User-level',
                'description': 'Establish network connection',
                'risk_level': 'Medium',
                'threat_assessment': 'Can be used to establish C2 connection',
                'attack_technique': 'C2 Communication',
                'mitre_technique': 'T1071',
                'malicious_score': 7.1
            },
            'ShellExecuteW': {
                'category': 'Process Control',
                'privilege_level': 'User-level',
                'description': 'Execute program or open file',
                'risk_level': 'High',
                'threat_assessment': 'Can be used to execute malicious programs',
                'attack_technique': 'Program Execution',
                'mitre_technique': 'T1059',
                'malicious_score': 8.3
            },
            'WinExec': {
                'category': 'Process Control',
                'privilege_level': 'User-level',
                'description': 'Execute program',
                'risk_level': 'High',
                'threat_assessment': 'Can be used to execute malicious programs',
                'attack_technique': 'Program Execution',
                'mitre_technique': 'T1059',
                'malicious_score': 8.1
            },
            'GetTempPathW': {
                'category': 'File Operation',
                'privilege_level': 'User-level',
                'description': 'Get temporary directory path',
                'risk_level': 'Low',
                'threat_assessment': 'Commonly used to create malicious files in temp directory',
                'attack_technique': 'File Operation',
                'mitre_technique': 'T1005',
                'malicious_score': 5.5
            },
            'CreateMutexW': {
                'category': 'Synchronization Object',
                'privilege_level': 'User-level',
                'description': 'Create mutex object',
                'risk_level': 'Low',
                'threat_assessment': 'Common method to prevent multiple instances',
                'attack_technique': 'Anti-Analysis',
                'mitre_technique': 'T1497',
                'malicious_score': 5.8
            }
        }
    
    def _load_critical_apis_data(self) -> Dict[str, Dict]:
        """Backward compatibility method"""
        return self._get_critical_apis()
    
    def convert_cape_to_markdown(self, cape_json_path: str, output_path: str = None) -> str:
        """Convert CAPE JSON directly to English Markdown report"""
        try:
            # Read original CAPE JSON
            with open(cape_json_path, 'r', encoding='utf-8') as f:
                cape_data = json.load(f)
            
            # Extract information directly from CAPE data
            sample_info = self._extract_sample_info(cape_data)
            process_info = self._extract_process_info(cape_data)
            network_info = self._extract_network_info(cape_data)
            api_info = self._extract_api_info(cape_data)
            sensitive_ops = self._analyze_sensitive_system_modifications(cape_data)
            
            # Generate report
            report_sections = []
            
            # 1. Report header
            report_sections.append(self._generate_header(sample_info))
            
            # 2. Sample basic information
            report_sections.append(self._generate_sample_overview(sample_info))
            
            # 3. Process behavior analysis
            if process_info['processes']:
                report_sections.append(self._generate_process_analysis(process_info))
            
            # 4. File and registry activity
            report_sections.append(self._generate_file_registry_analysis(cape_data))
            
            # 4.5 Sensitive System Operations
            if sensitive_ops:
                report_sections.append(self._generate_sensitive_operations_analysis(sensitive_ops))
            
            # 5. Network activity analysis
            if network_info['has_activity']:
                report_sections.append(self._generate_network_analysis(network_info))
            
            # 6. API analysis
            if api_info['all_apis']:
                report_sections.append(self._generate_api_analysis(api_info))
            
            # 7. Critical API analysis
            critical_apis = self._analyze_critical_apis(cape_data)
            report_sections.append(self._generate_critical_api_analysis(critical_apis))
            
            # Merge report
            markdown_report = "\n\n".join(report_sections)
            
            # Save report
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_report)
                print(f"✅ Report saved: {output_path}")
            
            return markdown_report
            
        except Exception as e:
            error_msg = f"❌ Conversion failed: {str(e)}"
            print(error_msg)
            return error_msg
    
    def _extract_sample_info(self, cape_data: Dict) -> Dict:
        """Extract sample information directly from CAPE data"""
        target_file = cape_data.get('target', {}).get('file', {})
        pe_info = target_file.get('pe', {})
        
        info = {
            'md5': target_file.get('md5', ''),
            'sha256': target_file.get('sha256', ''),
            'sha1': target_file.get('sha1', ''),
            'file_name': target_file.get('name', ''),
            'file_size': target_file.get('size', 0),
            'file_type': target_file.get('type', ''),
            'pe_info': {
                'entry_point': pe_info.get('entrypoint', ''),
                'image_base': pe_info.get('imagebase', ''),
                'timestamp': pe_info.get('timestamp', ''),
                'pdb_path': pe_info.get('pdbpath', ''),
                'imphash': pe_info.get('imphash', ''),
                'sections_count': len(pe_info.get('sections', [])),
                'imports_count': len(pe_info.get('imports', []))
            },
            'digital_signature': {
                'has_signature': len(pe_info.get('digital_signers', [])) > 0,
                'signers': pe_info.get('digital_signers', []),
                'signature_valid': pe_info.get('guest_signers', {}).get('aux_valid', False)
            },
            'selfextract': self._extract_selfextract_info(target_file)
        }
        
        return info
    
    def _extract_selfextract_info(self, target_file: Dict) -> Dict:
        """Extract self-extraction file information"""
        selfextract = target_file.get('selfextract', {})
        
        if not selfextract:
            return {'has_selfextract': False}
        
        overlay = selfextract.get('overlay', {})
        extracted_files = overlay.get('extracted_files', [])
        
        processed_files = []
        total_size = 0
        
        for file_info in extracted_files:
            if isinstance(file_info, dict):
                size = file_info.get('size', 0)
                if isinstance(size, (int, float)) and size > 0:
                    total_size += size
                
                processed_files.append({
                    'name': file_info.get('name', 'Unknown'),
                    'size': size,
                    'size_formatted': self._format_file_size(size),
                    'type': file_info.get('type', 'Unknown'),
                    'md5': file_info.get('md5', '')
                })
        
        return {
            'has_selfextract': True,
            'extracted_files': processed_files,
            'total_files': len(processed_files),
            'total_size': total_size,
            'total_size_formatted': self._format_file_size(total_size),
            'password_protected': bool(overlay.get('password', ''))
        }
    
    def _extract_process_info(self, cape_data: Dict) -> Dict:
        """Extract process information directly from CAPE data"""
        behavior = cape_data.get('behavior', {})
        processes = behavior.get('processes', [])
        
        if not processes:
            return {'processes': [], 'process_count': 0}
        
        processed_processes = {}
        
        for process in processes:
            process_name = process.get('process_name', '')
            process_id = process.get('process_id', 0)
            
            # Build process information
            proc_info = {
                'pid': process_id,
                'ppid': process.get('parent_id', 0),
                'process_name': process_name,
                'start_time': process.get('first_seen', ''),
                'module_path': process.get('module_path', ''),
                'command_line': self._extract_command_line(process),
                'total_api_calls': len(process.get('calls', [])),
                'depth_in_tree': 0,
                'api_calls': process.get('calls', [])
            }
            
            processed_processes[f"{process_name}_{process_id}"] = proc_info
        
        # Calculate process tree depth
        self._calculate_process_depth(processed_processes)
        
        return {
            'processes': processed_processes,
            'process_count': len(processed_processes)
        }
    
    def _extract_command_line(self, process: Dict) -> str:
        """Extract command line arguments"""
        environ = process.get('environ', {})
        command_line = environ.get('CommandLine', '')
        
        if not command_line:
            command_line = process.get('command_line', '')
        
        return command_line
    
    def _calculate_process_depth(self, processes: Dict):
        """Calculate process tree depth"""
        pid_to_proc = {}
        for proc_key, proc_data in processes.items():
            pid = proc_data['pid']
            pid_to_proc[pid] = proc_data
        
        def calculate_depth(pid, visited=None):
            if visited is None:
                visited = set()
            
            if pid in visited or pid not in pid_to_proc:
                return 0
            
            visited.add(pid)
            proc_data = pid_to_proc[pid]
            ppid = proc_data['ppid']
            
            if ppid == 0 or ppid == pid:
                proc_data['depth_in_tree'] = 0
                return 0
            
            parent_depth = calculate_depth(ppid, visited.copy())
            depth = parent_depth + 1
            proc_data['depth_in_tree'] = depth
            return depth
        
        for proc_data in processes.values():
            calculate_depth(proc_data['pid'])
    
    def _extract_network_info(self, cape_data: Dict) -> Dict:
        """Extract network information directly from CAPE data"""
        network_data = cape_data.get('network', {})
        
        # Filter sandbox IPs
        tcp_connections = [tcp for tcp in network_data.get('tcp', []) 
                          if not self._is_sandbox_ip(tcp.get('dst', ''))]
        
        udp_connections = [udp for udp in network_data.get('udp', []) 
                          if not self._is_sandbox_ip(udp.get('dst', ''))]
        
        http_requests = [http for http in network_data.get('http', []) 
                        if not self._is_sandbox_ip(http.get('host', ''))]
        
        dns_queries = network_data.get('dns', [])
        
        # Extract DNS queries from UDP
        dns_from_udp = []
        for udp in udp_connections:
            if udp.get('dport') == 53:
                dns_from_udp.append({
                    'server': udp.get('dst', ''),
                    'port': udp.get('dport', 53),
                    'type': 'UDP_DNS'
                })
        
        all_dns = list(dns_queries) + dns_from_udp
        
        return {
            'has_activity': bool(tcp_connections or udp_connections or http_requests or all_dns),
            'tcp_connections': tcp_connections,
            'udp_connections': udp_connections,
            'http_requests': http_requests,
            'dns_queries': all_dns,
            'network_stats': {
                'tcp_count': len(tcp_connections),
                'udp_count': len(udp_connections),
                'http_count': len(http_requests),
                'dns_count': len(all_dns)
            }
        }
    
    def _extract_api_info(self, cape_data: Dict) -> Dict:
        """Extract API information directly from CAPE data"""
        behavior = cape_data.get('behavior', {})
        summary = behavior.get('summary', {})
        processes = behavior.get('processes', [])
        
        # Get resolved_apis from summary (if exists)
        resolved_apis = summary.get('resolved_apis', [])
        
        # Extract API names from all process API calls
        all_apis = set()
        for process in processes:
            calls = process.get('calls', [])
            for call in calls:
                api_name = call.get('api', '')
                if api_name:
                    all_apis.add(api_name)
        
        all_apis_list = list(all_apis)
        
        return {
            'resolved_apis': resolved_apis,  # Keep backward compatibility
            'all_apis': all_apis_list,      # New: all API calls
            'total_apis': len(all_apis_list)
        }
    
    def _analyze_critical_apis(self, cape_data: Dict) -> Dict:
        """Analyze 21 critical high-risk APIs"""
        behavior = cape_data.get('behavior', {})
        processes = behavior.get('processes', [])
        
        found_apis = {}
        category_stats = {}
        
        for process in processes:
            calls = process.get('calls', [])
            for call in calls:
                api_name = call.get('api', '')
                
                if api_name in self.critical_apis_data:
                    api_info = self.critical_apis_data[api_name]
                    category = api_info['category']
                    
                    # Record API calls
                    if api_name not in found_apis:
                        found_apis[api_name] = {
                            'api_info': api_info,
                            'call_count': 0,
                            'processes': set()
                        }
                    
                    found_apis[api_name]['call_count'] += 1
                    found_apis[api_name]['processes'].add(f"{process.get('process_name', '')}_{process.get('process_id', 0)}")
                    
                    # Category statistics
                    if category not in category_stats:
                        category_stats[category] = {
                            'unique_apis': set(),
                            'total_calls': 0
                        }
                    
                    category_stats[category]['unique_apis'].add(api_name)
                    category_stats[category]['total_calls'] += 1
        
        # Convert set to list
        for api_name in found_apis:
            found_apis[api_name]['processes'] = list(found_apis[api_name]['processes'])
        
        for category in category_stats:
            category_stats[category]['unique_apis'] = list(category_stats[category]['unique_apis'])
            category_stats[category]['unique_count'] = len(category_stats[category]['unique_apis'])
        
        return {
            'found_apis': found_apis,
            'category_stats': category_stats,
            'total_critical_apis': len(found_apis),
            'total_calls': sum(api['call_count'] for api in found_apis.values())
        }
    
    def _is_sandbox_ip(self, ip: str) -> bool:
        """Check if it's a sandbox IP"""
        if not ip:
            return True
        
        sandbox_patterns = [
            '192.168.', '10.', '172.16.', '172.17.', '172.18.', '172.19.',
            '172.20.', '172.21.', '172.22.', '172.23.', '172.24.', '172.25.',
            '172.26.', '172.27.', '172.28.', '172.29.', '172.30.', '172.31.',
            '127.', '169.254.', '0.0.0.0'
        ]
        
        return any(str(ip).startswith(pattern) for pattern in sandbox_patterns)
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size"""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f}KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f}MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"
    
    # Report generation methods
    def _generate_header(self, sample_info: Dict) -> str:
        """Generate report header"""
        sha256 = sample_info.get('sha256', sample_info.get('md5', 'unknown'))
        return f"Sample ID: {sha256}"
    
    def _generate_sample_overview(self, sample_info: Dict) -> str:
        """Generate sample overview"""
        file_size = self._format_file_size(sample_info.get('file_size', 0))
        
        pe_info = sample_info.get('pe_info', {})
        timestamp = pe_info.get('timestamp', '')
        compile_time = f"Compile time: {timestamp}" if timestamp else "Compile time unknown"
        
        sections = [
            "## 📋 Sample Information",
            "",
            "### 🔍 File Information",
            f"- **SHA256 Hash**: `{sample_info.get('sha256', '')}`",
            f"- **File Size**: {file_size}",
            f"- **File Type**: {sample_info.get('file_type', 'Unknown type')}",
            f"- **{compile_time}**"
        ]
        
        # Add PDB path information
        pdb_path = pe_info.get('pdb_path', '')
        if pdb_path:
            sections.append(f"- **PDB Path**: `{pdb_path}`")
        
        # Add self-extraction information
        selfextract = sample_info.get('selfextract', {})
        if selfextract.get('has_selfextract'):
            sections.append("")
            sections.append(self._generate_selfextract_section(selfextract))
        
        return "\n".join(sections)
    
    def _generate_selfextract_section(self, selfextract: Dict) -> str:
        """Generate self-extraction file information"""
        total_files = selfextract.get('total_files', 0)
        total_size = selfextract.get('total_size_formatted', '0 bytes')
        password_protected = selfextract.get('password_protected', False)
        
        password_text = " (password protected)" if password_protected else ""
        
        sections = [
            "### 📦 Self-Extracting/Embedded Files",
            f"- **Contains embedded files**: ✅ Detected **{total_files}** extracted files{password_text}",
            f"- **Total file size**: {total_size}"
        ]
        
        extracted_files = selfextract.get('extracted_files', [])
        if extracted_files:
            # File type statistics
            type_stats = {}
            for file_info in extracted_files:
                file_type = file_info.get('type', 'Unknown')
                type_stats[file_type] = type_stats.get(file_type, 0) + 1
            
            sections.append("- **File type distribution**:")
            for file_type, count in type_stats.items():
                sections.append(f"  - {file_type}: {count} files")
            
            # Show first 5 files
            sections.append("- **Embedded file details**:")
            for i, file_info in enumerate(extracted_files[:5], 1):
                name = file_info.get('name', 'Unknown')
                size_formatted = file_info.get('size_formatted', 'Unknown')
                file_type = file_info.get('type', 'Unknown')
                
                if len(name) > 40:
                    display_name = name[:40] + "..."
                else:
                    display_name = name
                
                # Add icon based on file type
                if 'PE' in file_type or 'executable' in file_type.lower():
                    icon = "⚠️"
                elif 'archive' in file_type.lower():
                    icon = "📦"
                elif 'script' in file_type.lower():
                    icon = "📜"
                else:
                    icon = "📄"
                
                sections.append(f"  - {icon} **{display_name}** ({size_formatted}, {file_type})")
            
            if len(extracted_files) > 5:
                sections.append(f"  - ... {len(extracted_files) - 5} more files")
        
        return "\n".join(sections)
    
    def _generate_process_analysis(self, process_info: Dict) -> str:
        """Generate process behavior analysis"""
        processes = process_info['processes']
        process_count = process_info['process_count']
        
        # Calculate maximum depth
        max_depth = max(proc['depth_in_tree'] for proc in processes.values()) + 1
        
        sections = [
            "## 📊 Process Behavior Analysis",
            f"Process Count: {process_count} Tree Depth: {max_depth}",
            "",
            "### 🌲 Process Execution Tree (ASCII)",
            "```"
        ]
        
        # Generate ASCII process tree
        tree_lines = self._generate_process_tree(processes)
        sections.extend(tree_lines)
        sections.append("```")
        
        return "\n".join(sections)
    
    def _generate_process_tree(self, processes: Dict) -> List[str]:
        """Generate ASCII process tree"""
        tree_lines = []
        
        # Build PID mapping and parent-child relationships
        pid_to_proc = {}
        children_map = {}
        
        for proc_key, proc_data in processes.items():
            pid = proc_data['pid']
            ppid = proc_data['ppid']
            pid_to_proc[pid] = proc_data
            children_map.setdefault(ppid, []).append(pid)
        
        # Find root processes
        root_pids = [proc['pid'] for proc in processes.values() if proc['depth_in_tree'] == 0]
        if not root_pids and processes:
            root_pids = [list(processes.values())[0]['pid']]
        
        def build_tree(pid: int, depth: int = 0):
            if pid not in pid_to_proc:
                return
            
            proc_data = pid_to_proc[pid]
            name = proc_data['process_name']
            cmd = proc_data['command_line']
            total_api = proc_data['total_api_calls']
            
            indent = "  " * depth
            connector = "*" if depth == 0 else "+-"
            
            # Main line
            tree_lines.append(f"{indent}{connector} {name} [PID={pid}]")
            
            # Command line
            if cmd:
                display_cmd = cmd[:320] + "..." if len(cmd) > 220 else cmd
                tree_lines.append(f"{indent}   CMD: {display_cmd}")
            
            # Info line
            tree_lines.append(f"{indent}   INFO: API calls={total_api}")
            
            # Child processes
            for child_pid in sorted(children_map.get(pid, [])):
                build_tree(child_pid, depth + 1)
        
        for root_pid in sorted(root_pids):
            build_tree(root_pid)
        
        return tree_lines
    
    def _generate_file_registry_analysis(self, cape_data: Dict) -> str:
        """Generate file and registry activity analysis"""
        # Extract file and registry operations from behavior.summary
        behavior = cape_data.get('behavior', {})
        summary = behavior.get('summary', {})
        
        # Extract more detailed API calls from processes to get actual file and registry values
        processes = behavior.get('processes', [])
        
        # File operation statistics - CAPE uses files key containing all accessed files
        all_files = summary.get('files', [])
        file_ops = {
            'accessed': all_files,  # CAPE only has files array in summary, containing all accessed files
            'created': summary.get('file_created', []) or summary.get('files_created', []),
            'written': summary.get('file_written', []) or summary.get('files_written', []),
            'deleted': summary.get('file_deleted', []) or summary.get('files_deleted', []),
            'read': summary.get('file_read', []) or summary.get('files_read', []),
            'opened': summary.get('opened_files', [])
        }
        
        # Registry operation statistics
        reg_ops = {
            'keys_set': summary.get('write_keys', []) or summary.get('written_keys', []),
            'keys_deleted': summary.get('delete_keys', []) or summary.get('deleted_keys', []),
            'keys_read': summary.get('read_keys', []) or summary.get('queried_keys', []),
            'keys_opened': summary.get('opened_keys', [])
        }
        
        # Extract registry value information from API calls
        reg_values = []
        additional_files = []
        
        for process in processes[:3]:  # Only process first 3 processes to avoid excessive complexity
            calls = process.get('calls', [])
            
            for call in calls[:200]:  # Increase to 200 calls to get more information
                api = call.get('api', '')
                args = call.get('arguments', [])
                
                # File APIs - extract additional file paths
                if api in ['NtCreateFile', 'NtWriteFile', 'NtReadFile', 'CreateFileW', 'CreateFileA', 
                          'NtOpenFile', 'WriteFile', 'ReadFile', 'DeleteFileW', 'MoveFileW', 'CopyFileW']:
                    file_path = ''
                    for arg in args:
                        # Look for filename parameter - usually FileName in CAPE
                        if arg.get('name', '') in ['FileName', 'lpFileName', 'FilePath', 'lpExistingFileName', 'lpNewFileName']:
                            file_path = arg.get('value', '')
                            break
                    
                    if file_path and file_path not in all_files and file_path not in additional_files:
                        additional_files.append(file_path)
                
                # Registry APIs - extract key-value pair information
                elif api in ['RegSetValueExW', 'RegCreateKeyExW', 'RegDeleteKeyW', 'RegQueryValueExW',
                           'RegOpenKeyExW', 'NtSetValueKey', 'NtQueryValueKey']:
                    reg_key = ''
                    reg_value_name = ''
                    reg_value_data = ''
                    reg_value_type = ''
                    
                    for arg in args:
                        arg_name = arg.get('name', '')
                        arg_value = arg.get('value', '')
                        
                        # Skip handle values (usually hex values starting with 0x)
                        if arg_value and str(arg_value).startswith('0x') and len(str(arg_value)) <= 10:
                            continue
                        
                        # Extract registry key path - only accept meaningful paths
                        if arg_name in ['lpSubKey', 'KeyName'] and arg_value and '\\' in str(arg_value):
                            reg_key = arg_value
                        # Extract value name
                        elif arg_name in ['ValueName', 'lpValueName'] and arg_value:
                            reg_value_name = arg_value
                        # Extract value data
                        elif arg_name in ['lpData', 'Data', 'DataBuffer'] and arg_value:
                            reg_value_data = arg_value
                        # Extract value type
                        elif arg_name in ['Type', 'dwType'] and arg_value:
                            reg_value_type = arg_value
                    
                    # Build complete registry information - only add entries with real paths
                    if reg_key and '\\' in reg_key and not reg_key.startswith('0x'):
                        full_path = reg_key
                        if reg_value_name and reg_value_name != reg_key:
                            full_path = f"{reg_key}\\{reg_value_name}"
                        
                        reg_entry = {
                            'path': full_path,
                            'value_name': reg_value_name,
                            'value_data': reg_value_data,
                            'value_type': reg_value_type,
                            'api': api
                        }
                        
                        # Avoid adding duplicate registry entries
                        if not any(existing['path'] == full_path for existing in reg_values):
                            reg_values.append(reg_entry)
        
        # Generate report
        sections = [
            "## 📁 File and Registry Details",
            ""
        ]
        
        # File operations section
        total_file_ops = sum(len(ops) for ops in file_ops.values()) + len(additional_files)
        
        sections.extend([
            "### File Operations",
            f"- **Total**: {total_file_ops} operations"
        ])
        
        if total_file_ops > 0:
            # Statistics by type
            for op_type, files in file_ops.items():
                if files:
                    type_name = {
                        'accessed': 'Accessed',
                        'created': 'Created',
                        'written': 'Written',
                        'deleted': 'Deleted',
                        'read': 'Read',
                        'opened': 'Opened'
                    }.get(op_type, op_type)
                    
                    # Display file paths, prefer full paths
                    display_files = []
                    for file_path in files[:5]:
                        if len(file_path) > 60:
                            # Keep filename, shorten path
                            filename = os.path.basename(file_path)
                            path_part = os.path.dirname(file_path)
                            if len(path_part) > 30:
                                path_part = path_part[:27] + "..."
                            display_files.append(f"{path_part}\\{filename}")
                        else:
                            display_files.append(file_path)
                    
                    sections.append(f"- **{type_name}** ({len(files)}): {', '.join(display_files)}")
                    if len(files) > 5:
                        sections.append(f"  *...and {len(files)-5} other files*")
            
            # Show additional files found from API calls
            if additional_files:
                sections.append(f"- **Additional files discovered via API** ({len(additional_files)}):")
                for file_path in additional_files[:10]:
                    if len(file_path) > 70:
                        filename = os.path.basename(file_path)
                        path_part = os.path.dirname(file_path)
                        if len(path_part) > 40:
                            path_part = path_part[:37] + "..."
                        display_path = f"{path_part}\\{filename}"
                    else:
                        display_path = file_path
                    sections.append(f"  - {display_path}")
                if len(additional_files) > 10:
                    sections.append(f"  - *...and {len(additional_files)-10} other files*")
        else:
            sections.append("- No file operation records")
        
        sections.append("")
        
        # Registry operations section
        total_reg_ops = sum(len(ops) for ops in reg_ops.values()) + len(reg_values)
        
        sections.extend([
            "### Registry Operations",
            f"- **Total**: {total_reg_ops} operations"
        ])
        
        if total_reg_ops > 0:
            # Statistics by type
            for op_type, keys in reg_ops.items():
                if keys:
                    type_name = {
                        'keys_set': 'Set',
                        'keys_deleted': 'Deleted',
                        'keys_read': 'Queried',
                        'keys_opened': 'Opened'
                    }.get(op_type, op_type)
                    
                    # Simplify registry path display
                    simplified_keys = []
                    for key in keys[:8]:  # Increase display count
                        simplified = key.replace('HKEY_LOCAL_MACHINE\\', 'HKLM\\')
                        simplified = simplified.replace('HKEY_CURRENT_USER\\', 'HKCU\\')
                        if len(simplified) > 60:
                            simplified = simplified[:57] + "..."
                        simplified_keys.append(simplified)
                    
                    sections.append(f"- **{type_name}** ({len(keys)}):")
                    # Show more registry key details
                    for key in simplified_keys:
                        sections.append(f"  - {key}")
                    if len(keys) > 8:
                        sections.append(f"  - *...and {len(keys)-8} other keys*")
            
            # Show registry value details (including data)
            if reg_values:
                sections.append("- **Registry Value Details**:")
                for reg_entry in reg_values[:10]:  # Show first 10 values
                    path = reg_entry['path']
                    value_name = reg_entry['value_name']
                    value_data = reg_entry['value_data']
                    
                    # Simplify path display
                    simplified_path = path.replace('HKEY_LOCAL_MACHINE\\', 'HKLM\\')
                    simplified_path = simplified_path.replace('HKEY_CURRENT_USER\\', 'HKCU\\')
                    if len(simplified_path) > 50:
                        simplified_path = simplified_path[:47] + "..."
                    
                    # If there's value data, show key-value pair
                    if value_data and value_name:
                        # Limit value data length
                        if len(str(value_data)) > 50:
                            display_data = str(value_data)[:47] + "..."
                        else:
                            display_data = str(value_data)
                        sections.append(f"  - `{simplified_path}` = `{display_data}`")
                    else:
                        sections.append(f"  - `{simplified_path}`")
                
                if len(reg_values) > 10:
                    sections.append(f"  - *...and {len(reg_values)-10} other values*")
        else:
            sections.append("- No registry operation records")
        
        return "\n".join(sections)
    
    def _generate_network_analysis(self, network_info: Dict) -> str:
        """Generate network activity analysis"""
        stats = network_info['network_stats']
        
        sections = ["## 🌐 Network Activity Analysis"]
        
        # Protocol statistics
        protocol_stats = []
        if stats['tcp_count'] > 0:
            protocol_stats.append(f"**TCP**: {stats['tcp_count']} connections")
        if stats['udp_count'] > 0:
            protocol_stats.append(f"**UDP**: {stats['udp_count']} connections")
        if stats['http_count'] > 0:
            protocol_stats.append(f"**HTTP**: {stats['http_count']} requests")
        if stats['dns_count'] > 0:
            protocol_stats.append(f"**DNS**: {stats['dns_count']} queries")
        
        if protocol_stats:
            sections.append(f"Network protocol activity statistics: {', '.join(protocol_stats)}")
        
        # DNS query details
        dns_queries = network_info['dns_queries']
        if dns_queries:
            sections.append("")
            sections.append("### 🔍 DNS Query Details")
            sections.append("")
            sections.append("| No. | DNS Server | Port | Query Type | Domain | Response |")
            sections.append("|-----|------------|------|------------|--------|----------|")
            
            for i, dns in enumerate(dns_queries[:10], 1):
                if isinstance(dns, dict):
                    if dns.get('type') == 'UDP_DNS':
                        server = dns.get('server', 'N/A')
                        port = dns.get('port', 53)
                        query_type = 'UDP_DNS'
                        domain = 'N/A'
                        response = 'N/A'
                    else:
                        server = 'N/A'
                        port = 53
                        domain = dns.get('request', dns.get('query', 'N/A'))
                        query_type = dns.get('type', 'A')
                        response_data = dns.get('answers', [])
                        response = ', '.join([ans.get('data', '') for ans in response_data[:2]]) if response_data else 'N/A'
                else:
                    server = 'N/A'
                    port = 53
                    domain = str(dns)
                    query_type = 'A'
                    response = 'N/A'
                
                sections.append(f"| {i} | `{server}` | {port} | {query_type} | `{domain}` | {response} |")
            
            if len(dns_queries) > 10:
                sections.append(f"| ... | *{len(dns_queries)-10} more queries* | ... | ... | ... | ... |")
        
        # TCP connection details
        tcp_connections = network_info['tcp_connections']
        if tcp_connections:
            sections.append("")
            sections.append("### 🔗 TCP Connection Details")
            sections.append("")
            sections.append("| No. | Target IP | Target Port | Source Port | Protocol Service | Connection Time |")
            sections.append("|-----|-----------|-------------|-------------|------------------|-----------------|")
            
            for i, tcp in enumerate(tcp_connections[:10], 1):
                dst_ip = tcp.get('dst', 'N/A')
                dst_port = tcp.get('dport', 'N/A')
                src_port = tcp.get('sport', 'N/A')
                time_val = tcp.get('time', 0)
                
                time_str = f"{time_val:.2f}s" if isinstance(time_val, (int, float)) and time_val > 0 else "N/A"
                port_service = self._get_port_service(dst_port)
                
                sections.append(f"| {i} | `{dst_ip}` | {dst_port} | {src_port} | {port_service} | {time_str} |")
            
            if len(tcp_connections) > 10:
                sections.append(f"| ... | *{len(tcp_connections)-10} more connections* | ... | ... | ... | ... |")
        
        # HTTP request details
        http_requests = network_info['http_requests']
        if http_requests:
            sections.append("")
            sections.append("### 🌐 HTTP Request Details")
            sections.append("")
            sections.append("| No. | Method | Host | URI Path | User-Agent | Data Length |")
            sections.append("|-----|--------|------|----------|------------|-------------|")
            
            for i, http in enumerate(http_requests[:10], 1):
                method = http.get('method', 'GET')
                host = http.get('host', 'N/A')
                uri = http.get('uri', '/')
                user_agent = http.get('user_agent', 'N/A')
                data = http.get('data', '')
                
                # Truncate overly long fields
                if len(uri) > 40:
                    uri = uri[:40] + "..."
                if len(user_agent) > 25:
                    user_agent = user_agent[:25] + "..."
                
                data_len = len(data) if data else 0
                data_size = f"{data_len}B" if data_len > 0 else "None"
                
                sections.append(f"| {i} | **{method}** | `{host}` | `{uri}` | {user_agent} | {data_size} |")
            
            if len(http_requests) > 10:
                sections.append(f"| ... | *{len(http_requests)-10} more requests* | ... | ... | ... | ... |")
        
        sections.append("")
        return "\n".join(sections)
    
    def _get_port_service(self, port) -> str:
        """Get service corresponding to port"""
        port_services = {
            21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
            53: "DNS", 80: "HTTP", 110: "POP3", 143: "IMAP",
            443: "HTTPS", 993: "IMAPS", 995: "POP3S"
        }
        try:
            return port_services.get(int(port), "Unknown")
        except (ValueError, TypeError):
            return "Unknown"
    
    def _generate_api_analysis(self, api_info: Dict) -> str:
        """Generate API analysis - only show selected APIs"""
        all_apis = api_info['all_apis']
        total_apis = api_info['total_apis']
        
        # Filter selected APIs from all APIs
        selected_apis_found = []
        for api_name in all_apis:
            # Check if it's one of the selected APIs
            clean_api_name = api_name.split('.')[-1] if '.' in api_name else api_name
            if clean_api_name in self.critical_apis_data:
                selected_apis_found.append((api_name, clean_api_name))
        
        sections = [
            "## 🔧 Static API Analysis",
            "",
            "*Selected APIs found in PE file import table*",
            "",
            f"Total resolved APIs: **{total_apis}**, Analysis-relevant APIs: **{len(selected_apis_found)}**",
            ""
        ]
        
        if selected_apis_found:
            sections.append("| No. | API Function | Module | Function Description |")
            sections.append("|-----|--------------|--------|---------------------|")
            
            for i, (full_api_name, clean_api_name) in enumerate(selected_apis_found, 1):
                # Extract module name
                if '.' in full_api_name:
                    module = full_api_name.split('.')[0].lower().replace('.dll', '')
                else:
                    # Infer module based on API name
                    if clean_api_name.startswith(('Nt', 'Zw', 'Rtl')):
                        module = 'ntdll'
                    elif clean_api_name.startswith('Reg'):
                        module = 'advapi32'
                    elif clean_api_name.startswith('Crypt'):
                        module = 'advapi32'
                    elif any(clean_api_name.startswith(prefix) for prefix in ['Create', 'Virtual', 'Open', 'Get', 'Set']):
                        module = 'kernel32'
                    else:
                        module = 'unknown'
                
                # Get API information
                api_info_detail = self.critical_apis_data[clean_api_name]
                description = api_info_detail.get('description', 'API function')
                
                sections.append(f"| {i} | `{clean_api_name}` | {module} | {description} |")
        else:
            sections.append("None of the analysis-relevant APIs found")
        
        sections.append("")
        return "\n".join(sections)
    
    def _analyze_critical_apis(self, cape_data: Dict) -> Dict:
        """Analyze 21 critical high-risk APIs"""
        behavior = cape_data.get('behavior', {})
        processes = behavior.get('processes', [])
        
        found_apis = {}
        category_stats = {}
        
        for process in processes:
            calls = process.get('calls', [])
            for call in calls:
                api_name = call.get('api', '')
                
                if api_name in self.critical_apis_data:
                    api_info = self.critical_apis_data[api_name]
                    category = api_info['category']
                    
                    # Record API calls
                    if api_name not in found_apis:
                        found_apis[api_name] = {
                            'api_info': api_info,
                            'call_count': 0,
                            'processes': set()
                        }
                    
                    found_apis[api_name]['call_count'] += 1
                    found_apis[api_name]['processes'].add(f"{process.get('process_name', '')}_{process.get('process_id', 0)}")
                    
                    # Category statistics
                    if category not in category_stats:
                        category_stats[category] = {
                            'unique_apis': set(),
                            'total_calls': 0
                        }
                    
                    category_stats[category]['unique_apis'].add(api_name)
                    category_stats[category]['total_calls'] += 1
        
        # Convert set to list
        for api_name in found_apis:
            found_apis[api_name]['processes'] = list(found_apis[api_name]['processes'])
        
        for category in category_stats:
            category_stats[category]['unique_apis'] = list(category_stats[category]['unique_apis'])
            category_stats[category]['unique_count'] = len(category_stats[category]['unique_apis'])
        
        return {
            'found_apis': found_apis,
            'category_stats': category_stats,
            'total_critical_apis': len(found_apis),
            'total_calls': sum(api['call_count'] for api in found_apis.values())
        }
    
    def _is_sandbox_ip(self, ip: str) -> bool:
        """Check if it's a sandbox IP"""
        if not ip:
            return True
        
        sandbox_patterns = [
            '192.168.', '10.', '172.16.', '172.17.', '172.18.', '172.19.',
            '172.20.', '172.21.', '172.22.', '172.23.', '172.24.', '172.25.',
            '172.26.', '172.27.', '172.28.', '172.29.', '172.30.', '172.31.',
            '127.', '169.254.', '0.0.0.0'
        ]
        
        return any(str(ip).startswith(pattern) for pattern in sandbox_patterns)
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size"""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f}KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f}MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"
    
    def _generate_critical_api_analysis(self, critical_apis: Dict) -> str:
        """Generate 21 critical API analysis report"""
        found_apis = critical_apis['found_apis']
        category_stats = critical_apis['category_stats']
        total_apis = critical_apis['total_critical_apis']
        
        sections = [
            "## 🔑 Selected API Behavior Analysis",
            "",
            f"**{total_apis} selected APIs observed**",
            ""
        ]
        
        if not found_apis:
            sections.append("No selected APIs observed in execution")
            return "\n".join(sections)
        
        # Neutral list of observed selected APIs
        sections.append("### Observed Selected APIs")
        sections.append("")
        for api_name, api_info in found_apis.items():
            meta = api_info.get('api_info', {})
            category = meta.get('category', 'N/A')
            description = meta.get('description', 'API function')
            mitre = meta.get('mitre_technique', '')
            sections.append(f"**`{api_name}`** ({category})")
            sections.append(f"- **Description**: {description}")
            sections.append(f"- **Call count**: {api_info.get('call_count', 0)}")
            sections.append(f"- **Processes involved**: {', '.join(api_info.get('processes', []))}")
            if mitre:
                sections.append(f"- **MITRE technique**: {mitre}")
                sections.append("")
        
        # Category statistics
        if category_stats:
            sections.extend([
                "",
                "### 📊 Category Statistics",
                "",
                "| Category | API Count | Total Calls |",
                "|----------|-----------|-------------|"
            ])
            
            for category, stats in category_stats.items():
                api_count = stats.get('unique_count', len(stats.get('unique_apis', [])))
                total_calls = stats.get('total_calls', 0)
                sections.append(f"| {category} | {api_count} | {total_calls} |")
        
        return "\n".join(sections)

    def _analyze_sensitive_system_modifications(self, cape_data: Dict) -> List[Dict]:
        """Analyze file and registry operations against a knowledge base of sensitive patterns, considering operation type."""
        findings = []
        processes = cape_data.get('behavior', {}).get('processes', [])

        # Index processes by probable operation capability (writers/readers/deleters)
        writer_procs = {'registry': [], 'file': []}
        deleter_procs = {'registry': [], 'file': []}
        reader_procs = {'registry': [], 'file': []}

        for process in processes:
            pid = process.get('process_id')
            pname = process.get('process_name', 'UnknownProcess')

            for call in process.get('calls', []):
                api = call.get('api', '')
                raw_args = call.get('arguments', [])
                
                op_type_category = None
                op_type_specific = None
                path = None
                value_out = ''

                # Determine operation category (file/registry) and path
                if call.get('category') == 'file':
                    op_type_category = 'file'
                    path = self._extract_file_path_from_args(raw_args)
                elif call.get('category') == 'registry':
                    op_type_category = 'registry'
                    reg_key, reg_value_name, reg_value_data, _reg_type = self._extract_registry_fields_from_args(raw_args)
                    # Build a unified path representation
                    if reg_key and reg_value_name and reg_value_name != reg_key:
                        path = f"{reg_key}\\{reg_value_name}"
                    else:
                        path = reg_key or ''
                    value_out = reg_value_data or ''

                if not op_type_category:
                    # Continue to next call if not file/registry op
                    continue

                # Maintain process capability indices regardless of path extracted
                if api in self.api_op_mappings[op_type_category]['write']:
                    writer_procs[op_type_category].append((pname, pid, api))
                elif api in self.api_op_mappings[op_type_category]['delete']:
                    deleter_procs[op_type_category].append((pname, pid, api))
                elif api in self.api_op_mappings[op_type_category]['read']:
                    reader_procs[op_type_category].append((pname, pid, api))

                if not path:
                    continue

                # Determine specific operation type (write/read/delete)
                for op, apis in self.api_op_mappings[op_type_category].items():
                    if api in apis:
                        op_type_specific = op
                        break
                
                if not op_type_specific:
                    continue

                # Check against sensitive patterns
                for pattern, details in self.sensitive_patterns[op_type_category][op_type_specific].items():
                    if pattern.search(path):
                        findings.append({
                            'type': f"{op_type_category.capitalize()} {op_type_specific.capitalize()}",
                            'path': path,
                            'value': value_out,
                            'process_name': pname,
                            'pid': pid,
                            'api': api,
                            'description': details['description'],
                            'category_tag': details['category_tag'],
                            'category': 'Potentially Malicious'
                        })

                # Check against benign patterns
                for pattern, details in self.benign_patterns[op_type_category][op_type_specific].items():
                    if pattern.search(path):
                        findings.append({
                            'type': f"{op_type_category.capitalize()} {op_type_specific.capitalize()}",
                            'path': path,
                            'value': value_out,
                            'process_name': pname,
                            'pid': pid,
                            'api': api,
                            'description': details['description'],
                            'category_tag': details['category_tag'],
                            'category': 'Notable Benign Behavior'
                        })

        # Fallback enrichment from behavior.summary to catch paths missing in per-call args
        behavior = cape_data.get('behavior', {})
        summary = behavior.get('summary', {})

        # Registry summary keys
        reg_write_keys = summary.get('write_keys', []) or summary.get('written_keys', [])
        reg_delete_keys = summary.get('delete_keys', []) or summary.get('deleted_keys', [])
        reg_read_keys = summary.get('read_keys', []) or summary.get('queried_keys', [])

        def add_summary_findings(paths: List[str], category: str, op: str):
            for rk in paths or []:
                if not isinstance(rk, str):
                    continue
                # choose a likely source process; prefer writers/readers/deleters by op
                src = None
                if op == 'write' and writer_procs['registry']:
                    src = writer_procs['registry'][0]
                elif op == 'delete' and deleter_procs['registry']:
                    src = deleter_procs['registry'][0]
                elif op == 'read' and reader_procs['registry']:
                    src = reader_procs['registry'][0]
                pname, pid, api = (src if src else ('UnknownProcess', 0, 'summary'))

                # classify using patterns
                matched = False
                for pattern, details in self.sensitive_patterns['registry'][op].items():
                    if pattern.search(rk):
                        findings.append({
                            'type': f"Registry {op.capitalize()}",
                            'path': rk,
                            'value': '',
                            'process_name': pname,
                            'pid': pid,
                            'api': api,
                            'description': details['description'],
                            'category_tag': details['category_tag'],
                            'category': category
                        })
                        matched = True
                        break
                if not matched:
                    for pattern, details in self.benign_patterns['registry'][op].items():
                        if pattern.search(rk):
                            findings.append({
                                'type': f"Registry {op.capitalize()}",
                                'path': rk,
                                'value': '',
                                'process_name': pname,
                                'pid': pid,
                                'api': api,
                                'description': details['description'],
                                'category_tag': details['category_tag'],
                                'category': 'Notable Benign Behavior'
                            })
                            break

        add_summary_findings(reg_write_keys, 'Potentially Malicious', 'write')
        add_summary_findings(reg_delete_keys, 'Potentially Malicious', 'delete')
        add_summary_findings(reg_read_keys, 'Notable Benign Behavior', 'read')

        # File summary keys
        file_created = summary.get('file_created', []) or summary.get('files_created', [])
        file_written = summary.get('file_written', []) or summary.get('files_written', [])
        file_deleted = summary.get('file_deleted', []) or summary.get('files_deleted', [])
        file_read = summary.get('file_read', []) or summary.get('files_read', [])

        def add_file_summary_findings(paths: List[str], category: str, op: str):
            for fp in paths or []:
                if not isinstance(fp, str):
                    continue
                src = None
                if op == 'write' and writer_procs['file']:
                    src = writer_procs['file'][0]
                elif op == 'delete' and deleter_procs['file']:
                    src = deleter_procs['file'][0]
                elif op == 'read' and reader_procs['file']:
                    src = reader_procs['file'][0]
                pname, pid, api = (src if src else ('UnknownProcess', 0, 'summary'))

                matched = False
                for pattern, details in self.sensitive_patterns['file'][op].items():
                    if pattern.search(fp):
                        findings.append({
                            'type': f"File {op.capitalize()}",
                            'path': fp,
                            'value': '',
                            'process_name': pname,
                            'pid': pid,
                            'api': api,
                            'description': details['description'],
                            'category_tag': details['category_tag'],
                            'category': category
                        })
                        matched = True
                        break
                if not matched:
                    for pattern, details in self.benign_patterns['file'][op].items():
                        if pattern.search(fp):
                            findings.append({
                                'type': f"File {op.capitalize()}",
                                'path': fp,
                                'value': '',
                                'process_name': pname,
                                'pid': pid,
                                'api': api,
                                'description': details['description'],
                                'category_tag': details['category_tag'],
                                'category': 'Notable Benign Behavior'
                            })
                            break

        add_file_summary_findings(list(set((file_created or []) + (file_written or []))), 'Potentially Malicious', 'write')
        add_file_summary_findings(file_deleted, 'Potentially Malicious', 'delete')
        add_file_summary_findings(file_read, 'Notable Benign Behavior', 'read')

        return findings

    def _extract_file_path_from_args(self, args: Any) -> str:
        """Extract a meaningful file path from CAPE call arguments."""
        if isinstance(args, dict):
            candidates = [
                args.get('FileName'), args.get('lpFileName'), args.get('FilePath'),
                args.get('lpExistingFileName'), args.get('lpNewFileName'), args.get('NewFileName'), args.get('OldFileName'),
                args.get('filename'), args.get('filepath'), args.get('filepath_r'), args.get('oldpath'), args.get('newpath')
            ]
            for c in candidates:
                if c:
                    return str(c)
            return ''
        # CAPE commonly uses a list of {name, value}
        path = ''
        preferred_names = {
            'FileName', 'lpFileName', 'FilePath', 'lpExistingFileName', 'lpNewFileName', 'NewFileName', 'OldFileName',
            'filename', 'filepath', 'filepath_r', 'oldpath', 'newpath'
        }
        for arg in args or []:
            n = str(arg.get('name', ''))
            v = arg.get('value', '')
            if not v:
                continue
            if n in preferred_names:
                path = str(v)
                break
            # Some CAPE traces include ObjectAttributes with a nested path-like string
            if n == 'ObjectAttributes' and isinstance(v, str) and ('\\' in v or '/' in v):
                path = v
        return path

    def _extract_registry_fields_from_args(self, args: Any) -> 'Tuple[str, str, str, str]':
        """Extract registry key, value name, data and type from CAPE call arguments."""
        reg_key = ''
        reg_value_name = ''
        reg_value_data = ''
        reg_value_type = ''
        if isinstance(args, dict):
            # Rare shape, keep simple
            reg_key = str(args.get('lpSubKey') or args.get('KeyName') or args.get('SubKey') or '')
            reg_value_name = str(args.get('ValueName') or args.get('lpValueName') or '')
            reg_value_data = args.get('lpData') or args.get('Data') or args.get('DataBuffer') or ''
            reg_value_type = str(args.get('Type') or args.get('dwType') or '')
            return reg_key, reg_value_name, reg_value_data, reg_value_type
        for arg in args or []:
            name = str(arg.get('name', ''))
            value = arg.get('value', '')
            if value and isinstance(value, str) and value.startswith('0x') and len(value) <= 10:
                # ignore likely handle values
                continue
            if name in ['lpSubKey', 'KeyName', 'SubKey'] and value and '\\' in str(value):
                reg_key = str(value)
            elif name in ['ValueName', 'lpValueName'] and value:
                reg_value_name = str(value)
            elif name in ['lpData', 'Data', 'DataBuffer'] and value:
                reg_value_data = value
            elif name in ['Type', 'dwType'] and value:
                reg_value_type = str(value)
        return reg_key, reg_value_name, reg_value_data, reg_value_type

    def _generate_sensitive_operations_analysis(self, sensitive_ops: List[Dict]) -> str:
        """Generate Markdown section for sensitive file and registry operations."""
        sections = [
            "## 🔬 Classified System Operations",
            ""
        ]
        
        malicious_ops_by_cat = {}
        benign_ops_by_cat = {}

        for op in sensitive_ops:
            if op['category'] == 'Potentially Malicious':
                cat_tag = op.get('category_tag', 'Uncategorized')
                if cat_tag not in malicious_ops_by_cat:
                    malicious_ops_by_cat[cat_tag] = []
                malicious_ops_by_cat[cat_tag].append(op)
            else:
                cat_tag = op.get('category_tag', 'Uncategorized')
                if cat_tag not in benign_ops_by_cat:
                    benign_ops_by_cat[cat_tag] = []
                benign_ops_by_cat[cat_tag].append(op)


        if malicious_ops_by_cat:
            sections.append("### Potentially Suspicious Behaviors")
            sections.append("")
            for category, ops in sorted(malicious_ops_by_cat.items()):
                sections.append(f"#### {category}")
                for op in ops:
                    sections.append(f"- **Description:** {op['description']}")
                    details = f"`{op['path']}`"
                    if op.get('value'):
                        value_str = str(op['value'])
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        details += f" = `{value_str}`"
                    
                    sections.append(f"  - **Operation:** {op['type']} on {details}")
                    sections.append(f"  - **Source Process:** `{op['process_name']}` (PID: {op['pid']}) via API `{op['api']}`")
                    sections.append("")

        if benign_ops_by_cat:
            sections.append("### Notable Benign Behaviors")
            sections.append("")
            for category, ops in sorted(benign_ops_by_cat.items()):
                sections.append(f"#### {category}")
                for op in ops:
                    sections.append(f"- **Description:** {op['description']}")
                    details = f"`{op['path']}`"
                    if op.get('value'):
                        value_str = str(op['value'])
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        details += f" = `{value_str}`"
                    sections.append(f"  - **Details:** {op['type']} on {details}")
                    sections.append(f"  - **Source Process:** `{op['process_name']}` (PID: {op['pid']}) via API `{op['api']}`")
                    sections.append("")
        
        return "\n".join(sections)


def main():
    """Test English direct converter"""
    converter = DirectCapeConverterEnglish()
    
    # Test sample paths
    test_samples = [
        "/path/to/benign_samples/002cdf612509807b33e4ab09c686a966.json",
        "/path/to/malicious_samples/0000638ebbfe0d620abe6ca32abb1b58.json"
    ]
    
    output_dir = "/path/to/output/direct_conversion_test_english"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Testing English Direct CAPE Converter")
    print("=" * 50)
    
    for sample_path in test_samples:
        if os.path.exists(sample_path):
            sample_name = os.path.basename(sample_path).replace('.json', '')
            sample_type = 'benign' if 'benign' in sample_path else 'malicious'
            
            output_path = os.path.join(output_dir, f"{sample_name}_{sample_type}_english.md")
            
            print(f"\n📝 Converting sample: {sample_name}")
            
            start_time = datetime.now()
            result = converter.convert_cape_to_markdown(sample_path, output_path)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            if "❌" not in result:
                print(f"✅ Conversion successful, time: {duration:.2f}s")
                print(f"📄 Report length: {len(result)} characters")
            else:
                print(f"❌ Conversion failed: {result}")
        else:
            print(f"⚠️ Sample file not found: {sample_path}")
    
    print(f"\n📁 Test results saved in: {output_dir}")


if __name__ == "__main__":
    main()
