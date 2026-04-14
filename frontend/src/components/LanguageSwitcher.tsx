import { useTranslation } from 'react-i18next';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { ChevronUp } from "lucide-react";

const FlagUZ = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 250 125" className="h-4 w-6 rounded-sm object-cover shadow-sm">
        <rect width="250" height="125" fill="#ffffff" />
        <rect width="250" height="40" fill="#0099b5" />
        <rect y="85" width="250" height="40" fill="#1eb53a" />
        <rect y="40" width="250" height="2.5" fill="#ce1126" />
        <rect y="82.5" width="250" height="2.5" fill="#ce1126" />
        <circle cx="30" cy="20" r="12" fill="#ffffff" />
        <circle cx="34" cy="20" r="11" fill="#0099b5" />
        <g fill="#ffffff" transform="translate(18, 12) scale(0.6)">
            <g transform="translate(48, 4)">
                <polygon points="0,-6 2,0 -4,-4 4,-4 -2,0" />
                <polygon points="15,-6 17,0 11,-4 19,-4 13,0" />
                <polygon points="30,-6 32,0 26,-4 34,-4 28,0" />
            </g>
            <g transform="translate(40, 16)">
                <polygon points="0,-6 2,0 -4,-4 4,-4 -2,0" />
                <polygon points="15,-6 17,0 11,-4 19,-4 13,0" />
                <polygon points="30,-6 32,0 26,-4 34,-4 28,0" />
                <polygon points="45,-6 47,0 41,-4 49,-4 43,0" />
            </g>
            <g transform="translate(32, 28)">
                <polygon points="0,-6 2,0 -4,-4 4,-4 -2,0" />
                <polygon points="15,-6 17,0 11,-4 19,-4 13,0" />
                <polygon points="30,-6 32,0 26,-4 34,-4 28,0" />
                <polygon points="45,-6 47,0 41,-4 49,-4 43,0" />
                <polygon points="60,-6 62,0 56,-4 64,-4 58,0" />
            </g>
        </g>
    </svg>
);

const FlagRU = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 9 6" className="h-4 w-6 rounded-sm object-cover shadow-sm">
        <rect width="9" height="6" fill="#fff" />
        <rect y="2" width="9" height="4" fill="#E91E63" /> {/* Using pink-ish red for distinction, standard is d52b1e */}
        <rect y="2" width="9" height="2" fill="#0039A6" />
        <rect y="4" width="9" height="2" fill="#D52B1E" />
    </svg>
);

const FlagGB = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 30" className="h-4 w-6 rounded-sm object-cover shadow-sm">
        <clipPath id="s">
            <path d="M0,0 v30 h60 v-30 z" />
        </clipPath>
        <clipPath id="t">
            <path d="M30,15 h30 v15 z v15 h-30 z h-30 v-15 z v-15 h30 z" />
        </clipPath>
        <g clipPath="url(#s)">
            <path d="M0,0 v30 h60 v-30 z" fill="#012169" />
            <path d="M0,0 L60,30 M60,0 L0,30" stroke="#fff" strokeWidth="6" />
            <path d="M0,0 L60,30 M60,0 L0,30" clipPath="url(#t)" stroke="#C8102E" strokeWidth="4" />
            <path d="M30,0 v30 M0,15 h60" stroke="#fff" strokeWidth="10" />
            <path d="M30,0 v30 M0,15 h60" stroke="#C8102E" strokeWidth="6" />
        </g>
    </svg>
);

const LANGUAGES = [
    { code: 'uz', label: "O'zbek", flag: <FlagUZ /> },
    { code: 'ru', label: 'Русский', flag: <FlagRU /> },
    { code: 'en', label: 'English', flag: <FlagGB /> }
];

export function LanguageSwitcher() {
    const { i18n } = useTranslation();

    const changeLanguage = (lng: string) => {
        i18n.changeLanguage(lng);
    };

    const currentLang = LANGUAGES.find(l => l.code === i18n.language) || LANGUAGES[0];

    return (
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <Button
                    variant="ghost"
                    className="w-full justify-between px-3 h-10 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground group"
                >
                    <span className="flex items-center gap-2">
                        <span className="flex items-center justify-center w-6">{currentLang.flag}</span>
                        <span className="text-sm font-medium">{currentLang.label}</span>
                    </span>
                    <ChevronUp className="h-4 w-4 text-muted-foreground group-hover:text-foreground transition-colors" />
                </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-[200px] mb-2">
                {LANGUAGES.map((lang) => (
                    <DropdownMenuItem
                        key={lang.code}
                        onClick={() => changeLanguage(lang.code)}
                        className="cursor-pointer flex items-center gap-2 py-2.5"
                    >
                        <span className="flex items-center justify-center w-6">{lang.flag}</span>
                        <span className="text-sm font-medium">{lang.label}</span>
                        {i18n.language === lang.code && (
                            <div className="ml-auto w-1.5 h-1.5 rounded-full bg-primary" />
                        )}
                    </DropdownMenuItem>
                ))}
            </DropdownMenuContent>
        </DropdownMenu>
    );
}
